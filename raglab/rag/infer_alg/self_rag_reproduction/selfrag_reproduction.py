
from typing import Optional, Any
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pdb
import pudb
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
from raglab.rag.infer_alg.self_rag_reproduction.utils import load_special_tokens, postprocess_answer_option_conditioned, preprocess_input_data
from raglab.rag.infer_alg.self_rag_reproduction.utils import PROMPT_DICT, TASK_INST,process_data_evidences, postprocess, fix_spacing
from raglab.dataset.utils import get_dataset
from raglab.dataset.base_dataset import MultiChoiceQA

class SelfRag_Reproduction(NaiveRag):
    def __init__(self, args):
        super().__init__(args) # define common args, setup_retrieval, load_llm()
        self.init(args)
    
    def init(self, args):
        # load llm args
        self.download_dir = args.download_dir
        self.world_size = args.world_size
        self.dtype = args.dtype

        # decoding args
        self.threshold = args.threshold
        self.use_seqscore = args.use_seqscore
        self.use_groundness = args.use_groundness
        self.use_utility = args.use_utility
        self.w_rel = args.w_rel
        self.w_sup = args.w_sup
        self.w_use = args.w_use
        self.beam_width = args.beam_width
        self.max_depth = args.max_depth

        # retrieval 
        self.retrieval_mode = args.retrieval_mode
        self.show_specialtokens = args.show_specialtokens
        self.realtime_retrieval = args.realtime_retrieval
        self.inference_form = args.inference_form
        self.ignore_cont = args.ignore_cont
        self.use_citation = args.use_citation
        assert self.beam_width < self.n_docs, "The beam width should be less than the number of documents."

    def inference(self, query: Optional[str]=None, mode='interact', task=None):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            input = f"### Instruction:\n{query}\n\n### Response:\n"
            source_question = query
            pregiven_passages = []
            if 'short_form' == self.inference_form:
                final_prediction, generation_track = self.short_form_generation(input, source_question, pregiven_passages,
                                                            use_seqscore = self.use_seqscore, threshold = self.threshold,
                                                            w_rel = self.w_rel, w_sup = self.w_sup, w_use = self.w_use, mode = mode)
                catation_docs = {0:None}
                return final_prediction, catation_docs, generation_track
            elif 'long_form' == self.inference_form:
                final_prediction, generation_track = self.long_form_generation(input, source_question, pregiven_passages, 
                                                            beam_width=self.beam_width, max_depth=self.max_depth, 
                                                            w_rel=self.w_rel, w_sup=self.w_sup, w_use=self.w_use, 
                                                            use_seqscore=self.use_seqscore,ignore_cont=self.ignore_cont)
                final_prediction_with_citation, catation_docs = self.aggregate_response_with_citation(final_prediction, generation_track, add_citation=self.use_citation)      
                return  final_prediction_with_citation, catation_docs, generation_track
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset()
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for instance_idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                eval_data = self.EvalData.preprocess(eval_data)
                input = self.EvalData.get_instruction(eval_data[self.EvalData.inputStruction.question])
                source_question = eval_data[self.EvalData.inputStruction.question]
                if self.realtime_retrieval == True:
                    pregiven_passages = []
                else:
                    # get the pregiven passages form local eval datasets
                    pregiven_passages = eval_data[self.EvalData.inputStruction.pregiven_passages][:self.n_docs]
                if 'short_form' == self.inference_form:
                    final_prediction, generation_track = self.short_form_generation(input, source_question, pregiven_passages,
                                                                            use_seqscore = self.use_seqscore, threshold = self.threshold,
                                                                            w_rel = self.w_rel, w_sup = self.w_sup, w_use = self.w_use, mode = mode)
                    if "SUPPORTS" in final_prediction: # In some situation LLM will generate SUPPORTS or REFUTES instead of true or flase
                        final_prediction = "true" 
                    elif "REFUTES" in final_prediction:
                        final_prediction = "false"
                    inference_results = self.EvalData.record_result(eval_data, final_prediction, inference_results)
                    eval_result = self.EvalData.eval_acc(inference_results)
                    print(f'{self.task} Accuracy in {instance_idx} turn: {eval_result}')
                elif 'long_form' == self.inference_form:
                    final_prediction, generation_track = self.long_form_generation(input, source_question, pregiven_passages, 
                                                                beam_width=self.beam_width, max_depth=self.max_depth, 
                                                                w_rel=self.w_rel, w_sup=self.w_sup, w_use=self.w_use, 
                                                                use_seqscore=self.use_seqscore,ignore_cont=self.ignore_cont)
                    final_prediction_with_citation, catation_docs = self.aggregate_response_with_citation(final_prediction, generation_track, add_citation=self.use_citation)  
                    response_id = 0
                    inference_results = self.EvalData.record_result(eval_data, final_prediction_with_citation,
                                                                    catation_docs, response_id, generation_track,
                                                                    inference_results)
                    # end of for loop
            self.EvalData.save_result(inference_results) 
            if 'short' == self.inference_form:
                eval_result = self.EvalData.eval_acc(inference_results)
                print(f'Final {self.task} accuracy: {eval_result}')
                return eval_result
            elif 'long' == self.inference_form:
                return 'Inference completion'
    
    def load_llm(self):
        llm = LLM(model=self.llm_path, dtype=self.dtype)
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_path, padding_side="left")
        return llm, tokenizer, sampling_params

    def short_form_generation(self, prompt:str, source_question:str, pregiven_passages:Optional[None],
                            use_seqscore=True, threshold=0.2,w_rel=1.0, w_sup=1.0, w_use=0.5, mode = 'evaluation'): 

        retrieval_tokens, relevant_tokens, ground_tokens, utility_tokens = load_special_tokens(
                                self.tokenizer, use_grounding=self.use_groundness, use_utility=self.use_utility)
        generation_track = {}
        if 'always_retrieval' == self.retrieval_mode:
            do_retrieve = True
        elif 'no_retrieval' == self.retrieval_mode:
            do_retrieve = False
        elif 'adaptive_retrieval' == self.retrieval_mode:
            #retrieval or not base on first token
            ratio, generation_track = self.firstToken_retrievalRatio(prompt, retrieval_tokens, generation_track)
            do_retrieve = ratio > threshold
        # "do retrieval or not retrieval
        if do_retrieve is True:   
            if self.realtime_retrieval == True:
                passages = self.retrieval.search(source_question)
                evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(passage["title"], passage["content"]) for rank, passage in passages.items()] 
            else:
                evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(para["title"], para["text"]) for para in pregiven_passages] 
            preds = self.llm.generate(evidence_augmented_inputs, self.sampling_params)
            # calculate score of each candidate
            relevance_score_dict = {}
            grd_score_dict = {}
            ut_score_dict = {}
            overall_scores = {}
            for p_idx, pred in enumerate(preds): 
                #sequence score 
                seq_score = self.sequence_score(pred)
                # init dict in each loop
                relevance_score_dict.setdefault(p_idx, {}) 
                grd_score_dict.setdefault(p_idx, {})
                ut_score_dict.setdefault(p_idx, {})
                # relevance score 
                relevance_score, relevance_score_dict = self.relevanceToken_score(pred, relevant_tokens, p_idx, relevance_score_dict)
                # Issupport score
                ground_score, grd_score_dict = self.IssupportToken_score(pred, ground_tokens, p_idx, grd_score_dict)
                # Utility score
                utility_score, ut_score_dict = self.UtilityToken_score(pred, utility_tokens, p_idx, ut_score_dict)
                if use_seqscore is True:
                    final_score = seq_score + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
                else:
                    final_score = w_rel * relevance_score +  w_sup * ground_score + w_use * utility_score
                overall_scores[p_idx] = {"final_score": final_score,
                                        "relevance_score": relevance_score,
                                        "ground_score": ground_score,
                                        "utility_score": utility_score,
                                        "relevance_score_dict": relevance_score_dict, 
                                        "grd_score_dict": grd_score_dict,
                                        "ut_score_dict": ut_score_dict} #TODO 考虑删除这段代码的必要性
                pred_text = pred.outputs[0].text
                if self.realtime_retrieval == True:
                    generation_track["retrieval_{}".format(p_idx+1)] = {"pred": pred_text, "score": float(final_score), "ctx": passages[p_idx+1]}
                else:
                    generation_track["retrieval_{}".format(p_idx+1)] = {"pred": pred_text, "score": float(final_score), "ctx": pregiven_passages[p_idx]}
        else: 
            # no retrieval generation
            prompt += "[No Retrieval]"
            preds = self.llm.generate([prompt], self.sampling_params)
            pred = preds[0].outputs[0].text 
            generation_track['no_retrieval'] = {"pred": pred} # no retrieval no need score and passages
        
        # Aggregating answers
        if len(generation_track) <= 2: 
            # post process for no retrieval
            if True == self.show_specialtokens:
                return pred, generation_track
            else:
                # remove all sprcial tokens 
                postprocessed_pred = postprocess_answer_option_conditioned(pred) 
                return postprocessed_pred, generation_track 
        else:
            answer2score = {}
            if 'evaluation' == mode and isinstance(self.EvalData, MultiChoiceQA) == True:
                '''
                Aggregating for multi-choice question
                source explaination: For SELF-RAG inference on PubHealth and ARC-C, instead of determining the output with the highest score as in other tasks, 
                                    we aggregate the scores for each option and select the answer option with the highest score.       
                paper: https://arxiv.org/abs/2310.11511
                '''
                for key, result in generation_track.items():
                    if key == "decide_retrieval_mode":
                        continue
                    answer = postprocess_answer_option_conditioned(result["pred"])
                    score = result["score"]
                    answer2score.setdefault(answer, 0)
                    answer2score[answer] += score
                sorted_answers = sorted(
                    answer2score.items(), key=lambda x: x[1], reverse=True)
                best_option = sorted_answers[0][0]
            else:
                path2score = {key: item["score"] for key, item in generation_track.items() if key != "decide_retrieval_mode"} 
                best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][0]
                best_option = generation_track[best_path]["pred"]
                if self.show_specialtokens == True:
                    pass
                else:
                    # remove all special token 
                    best_option = postprocess_answer_option_conditioned(best_option)
        return best_option, generation_track 

    def long_form_generation(self, prompt: str, source_question: str, pregiven_passages:Optional[dict],
                             beam_width=2, max_depth=7,w_rel=1.0, w_sup=1.0, w_use=0.5, 
                             use_seqscore = True,ignore_cont = None) -> tuple[dict[int,str], dict, bool]: 

        retrieval_tokens, relevant_tokens, ground_tokens, utility_tokens = load_special_tokens(self.tokenizer, 
                                                                            use_grounding=self.use_groundness, 
                                                                             use_utility=self.use_utility)

        if 'no_retrieval' == self.retrieval_mode:
            prompt += "[No Retrieval]" 
            preds = self.llm.generate([prompt], self.sampling_params)
            preds_text = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
            final_prediction = {0:preds_text[0]} 
            generation_track = {"original_splitted_sentences": {0:preds_text}}
            return final_prediction, generation_track
        elif 'always_retrieval' == self.retrieval_mode:
            '''
            Diff: The logic of adaptive retrieval is based on paper and GitHub issue, which is different from the self RAG source code (run_long_form_static.py). 
            Raglab has truly implemented the multi-turn retrieval proposed in the self rag paper for the first time.
            '''
            curr_depth = 1
            node_id = 0
            prediction_tree = {}
            levels = {}
            prediction_tree[node_id] = {"prompt": prompt, "pred": "", 
                                        "processed_pred": "", "score": None, "ctx": None, "parent": None} # [First retrieve flag] means 
            levels[0] = [0]
            while curr_depth < max_depth:
                # bulid tree
                levels[curr_depth]= []
                if curr_depth - 1 in levels:
                    for parent_node in levels[curr_depth-1]:
                        # 对于当前来说当curr_depth == 2 的时候parent_node确实是 1，也就是从 1 里面拿出来存储的东西
                        prev_pred, prompt, prev_generation, prev_score = self.get_lastTurn_generation(parent_node, prediction_tree)
                        # retrirval model的 input 是不需要 special token
                        curr_prompt = prompt + prev_generation
                        # self rag input maintain special token
                        previous_sentence = postprocess(prev_pred) 
                        current_retrieval_input = source_question + previous_sentence
                        '''
                        这里是按照樱花妹论文里面写的方法实现的，每一次 retrieval 的 input 都是 source qeustion + 上一次生成的句子，并且按照道理讲，这个句子应该是remove special tokens 的句子，这样再检索的时候才能更加的准确
                        '''
                        curr_preds, curr_scores, overall_score_dict, retrieval_docs = self.run_step_generation_batch(curr_prompt, current_retrieval_input , pregiven_passages,
                                                                                                                     retrieval_tokens=retrieval_tokens, relevant_tokens=relevant_tokens,
                                                                                                                     ground_tokens=ground_tokens, utility_tokens=utility_tokens,
                                                                                                                     w_rel=w_rel, w_sup=w_sup, w_use=w_use, use_seqscore=use_seqscore)
                        
                        prediction_tree, node_id, levels = self.set_predictionTree(curr_depth, parent_node, node_id, 
                                                                           curr_preds, curr_scores, curr_prompt,
                                                                           prev_score, retrieval_docs, prediction_tree, levels ,overall_score_dict)
                    # end of the for loop 
                    current_rank = levels[curr_depth]
                    #get the top-2 score 
                    node2score = {node_id: prediction_tree[node_id]['score'] for node_id in current_rank} #
                    top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[:beam_width] # 取 top2 结果
                    levels[curr_depth] = [node[0] for node in top_nodes] 
                    curr_depth += 1  
                else:
                    break
            # end of the while curr_depth < max_depth:
            best_selections = self.backtracking_prediction_tree(levels, curr_depth, prediction_tree)
            # get final_prediction
            final_prediction = {}
            splitted_sentences = {}
            original_splitted_sentences = {}
            ctxs = {}
            for path_i, nodes in best_selections.items():
                final_prediction[path_i] = " ".join([prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))])
                splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
                original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

                ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
                    ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
            
            generation_track = {"final_prediction": final_prediction,
                    "splitted_sentences": splitted_sentences,
                    "original_splitted_sentences": original_splitted_sentences,
                    "best_selections": best_selections,
                    "ctxs": ctxs,
                    "prediction_tree": prediction_tree}
        
            return final_prediction, generation_track
        # the of elif 'always_retrieval' == self.retrieval_mode:
        elif 'adaptive_retrieval' == self.retrieval_mode:
            '''
            diff: The logic of adaptive retrieval is based on paper and GitHub issue, which is different from the self RAG source code (run_long_form_static.py). 
            Raglab has truly implemented the multi-turn retrieval proposed in the self rag paper for the first time.
            '''
            curr_depth = 1
            node_id = 0
            prediction_tree = {}
            levels = {}
            prediction_tree[node_id] = {"prompt": prompt, "pred": "", 
                                        "processed_pred": "", "score": None, "ctx": None, "parent": None} # [First retrieve flag] means 
            levels[0] = [0]
            level_tmp = [] # level_tmp is used to store the node when [No retrieval], and then after the entire tree is maintained, all nodes in level_tmp are merged into the tree. 
            while curr_depth < max_depth:
                # bulid tree
                if curr_depth - 1 in levels and len(levels[curr_depth-1])!=0: #nnd 应该是不等于0，等于 0 直接就是空的了
                    levels[curr_depth]= []
                    for parent_node in levels[curr_depth-1]:
                        prev_pred, prompt, prev_generation, prev_score = self.get_lastTurn_generation(parent_node, prediction_tree)

                        curr_prompt = prompt + prev_generation
                        # BUG Meet，讨论一下curr_prompt 和 current_retrieval_input 是什么
                        previous_sentence = postprocess(prev_pred) 
                        current_retrieval_input = source_question + previous_sentence
                        '''
                        这里是按照樱花妹论文里面写的方法实现的，每一次 retrieval 的 input 都是 source qeustion + 上一次生成的句子，并且按照道理讲，这个句子应该是remove special tokens 的句子，这样再检索的时候才能更加的准确
                        '''
                        ratio, _ = self.firstToken_retrievalRatio(curr_prompt, retrieval_tokens, None)
                        if ratio > self.threshold:
                            curr_preds, curr_scores, overall_score_dict, retrieval_docs = self.run_step_generation_batch(curr_prompt, current_retrieval_input , pregiven_passages,
                                                                                                        retrieval_tokens=retrieval_tokens, relevant_tokens=relevant_tokens, 
                                                                                                        ground_tokens=ground_tokens, utility_tokens=utility_tokens,
                                                                                                        w_rel=w_rel, w_sup=w_sup, w_use=w_use, use_seqscore=use_seqscore)
                            prediction_tree, node_id, levels = self.set_predictionTree(curr_depth, parent_node, node_id,curr_preds, 
                                                                                       curr_scores, curr_prompt,prev_score, retrieval_docs, 
                                                                                       prediction_tree, levels ,overall_score_dict)
                        else:
                            curr_preds, curr_scores, overall_score_dict, retrieval_docs = self.generation_without_retrieval(curr_prompt)
                            prediction_tree, node_id, level_tmp = self.set_predictionTree_NoRetrieval(curr_depth, parent_node, node_id, curr_preds, 
                                                                                                    curr_scores, curr_prompt, retrieval_docs, 
                                                                                                    prediction_tree, overall_score_dict, level_tmp)

                    # end of the for loop 
                    current_rank = levels[curr_depth] 
                    #get the top-k node based on sentence final score
                    node2score = {node_id: prediction_tree[node_id]['score'] for node_id in current_rank} #
                    top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True) # 
                    top_nodes = top_nodes[:(beam_width - len(level_tmp))] 
                    levels[curr_depth] = [node[0] for node in top_nodes] 
                    curr_depth += 1
                else:
                    break
            # end of the while curr_depth < max_depth:
            # Complete the tree(variable:levels)
                # The purpose of below snippet code is only to complete the logic of building tree(variable:levels), and it is not helpful for building the best response
            for no_retrieval_node in level_tmp:
                depth = no_retrieval_node['curr_depth']
                node_id = no_retrieval_node['node_id']
                levels[depth] = levels[depth] + [node_id]
            
            # {0: [0], 1: [1, 3], 2: [9, 11], 3: [15], 4: [19], 5: [23], 6: [28]}

            # backtraking the levels get the best answer
            best_selections = self.backtracking_prediction_tree(levels, curr_depth, prediction_tree)
            if len(best_selections) < self.beam_width:
                # get the last path_id in best_selections
                for path_id, best_selection in best_selections.items():
                    path_id = path_id
                
                best_selections = self.backtracking_prediction_tree_noRetrieval(best_selections, prediction_tree, level_tmp, path_id)

            # get final_prediction
            final_prediction = {}
            splitted_sentences = {}
            original_splitted_sentences = {}
            ctxs = {}
            for path_i, nodes in best_selections.items():
                final_prediction[path_i] = " ".join([prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))])
                splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
                original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

                ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
                    ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
            
            generation_track = {"final_prediction": final_prediction,
                    "splitted_sentences": splitted_sentences,
                    "original_splitted_sentences": original_splitted_sentences,
                    "best_selections": best_selections,
                    "ctxs": ctxs,
                    "prediction_tree": prediction_tree}
        
            return final_prediction, generation_track
    
    def generation_without_retrieval(self, prompt):
        '''
        # without retrieval and retruen one response
        '''
        prompt += "[No Retrieval]" 
        preds = self.llm.generate([prompt], self.sampling_params)
        curr_prediction = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        scores = [1] # The score of [No retrieval] output is 1. And the [No retrieval] outputs will not be sorted by score in rank process
        overall_scores = {0:None}
        retrieval_docs = {1:None}
        return curr_prediction, scores, overall_scores, retrieval_docs
    
    def aggregate_response_with_citation(self, final_predictions: dict[int,str], generation_track: dict[str, Any], add_citation = True)-> tuple[dict, dict]:
        '''
        # Aggregate response for response. If the response generate by no_retrieval mode. There is no need to add citation. 
        '''
        output_with_citation = {}
        catation_doc = {}
        for response_idx, generated_response in final_predictions.items(): 
            final_output = ""
            docs = []
            previous_generations = [] 
            if "splitted_sentences" not in generation_track:
                output_with_citation[response_idx] = fix_spacing(postprocess(generated_response))
                catation_doc[response_idx] = docs
            else:
                if len(postprocess(generated_response)) == 0:
                    generation_track["splitted_sentences"][response_idx], generation_track["ctxs"][response_idx] = generation_track["splitted_sentences"][response_idx], generation_track["ctxs"][response_idx] 
                # 上面这条我感觉大概率也不会用到
                for cite_idx, (sentence, doc) in enumerate(iterable=zip(generation_track["splitted_sentences"][response_idx], generation_track["ctxs"][response_idx])):
                    if len(sentence) == 0:
                        continue
                    sentence = postprocess(sentence) 
                    # remove the loopping sentence
                    if sentence in previous_generations: 
                        continue
                    else:
                        previous_generations.append(sentence)

                    if add_citation == True and doc is not None: 
                        sentence = sentence.replace(".[Continue to Use Evidence]", f" [{cite_idx}]. ")
                        final_output += sentence[:-1] + " [{}]".format(cite_idx) + ". " 
                        final_output = final_output.replace(f". [{cite_idx}] ", f" [{cite_idx}]. ")
                        docs.append(doc) # docs -> list[dict]
                        '''
                        # Diff: selfrag_reproduction.py implements the citation function under multiple rounds of retrieval, which is different from the logic of selfrag_orignal.py                        
                        '''
                    else:
                        final_output += sentence
                        # [No retrieval] do not cite doc
                if len(final_output) == 0:
                    final_output = fix_spacing(final_output)  
                if len(final_output) > 0 and final_output[-1] == " ":
                    final_output = final_output[:-1]
                final_output = fix_spacing(final_output)
                output_with_citation[response_idx] = final_output
                catation_doc[response_idx] = docs
        # end of the for loop
        return output_with_citation, catation_doc

    def backtracking_prediction_tree(self, levels: dict[int,list[int]], curr_depth: int, prediction_tree: dict[int, dict]) -> dict[int,list[int]]:
        '''
        get best tracking from prediction_tree base on levels
        '''
        parent = 0 
        best_selections = {}
        # Traverse from the bottom 
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0} # remove empty list in levels
        for path_i, node in enumerate(levels[len(levels)]): # beam search 
            if node == 0:
                break
            best_selections[path_i] = [node] 
            current_node = node 
            current_level = curr_depth 
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]["parent"]
                best_selections[path_i] = [parent] + best_selections[path_i] 
                current_node = parent 
                current_level -= 1
        
        return best_selections

    def backtracking_prediction_tree_noRetrieval(self, best_selections:dict[int,list], prediction_tree, level_tmp:list[dict], next_path_id:int):
        '''
        # back tracking the node in level_tmp 
        - level_tmp is generated by [No retrieval] inference process
        - this function is only used in adaptive retrieval in long form inference
        '''
        for no_retrieval_node in level_tmp:
            curr_depth = no_retrieval_node['curr_depth']
            node = no_retrieval_node['node_id']
            next_path_id += 1
            best_selections[next_path_id] = [node]
            current_node = node 
            current_level = curr_depth 
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]['parent']
                best_selections[next_path_id] = [parent] + best_selections[next_path_id]
                current_node = parent
                current_level -= 1
        return best_selections
    

    def set_predictionTree(self, curr_depth, parent_node, node_id:int,  curr_preds:list[str], curr_scores:list[float], curr_prompt:str, prev_score, retrieval_docs, prediction_tree, levels , overall_score_dict):
        retrieval_results = {}
        for i, (curr_pred, p_score) in enumerate(zip(curr_preds, curr_scores)): 
            retrieval_results[i] = {"pred": curr_pred, "score": p_score}
        for i, result in retrieval_results.items(): 
            node_id += 1 
            node_score = result["score"] * prev_score if prev_score is not None else result["score"]
            curr_pred = result["pred"] 
            if self.realtime_retrieval == True:
                # the index of real time retrieved passages begin from 1, but the index of pre-given passages begin from 0.
                prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred,
                                            "score": node_score, "ctx": retrieval_docs[i+1], "parent": parent_node,
                                            "overall_score_dict": overall_score_dict} # TODO 后续评估一下overall_score_dict的用处，貌似不需要保存下来
            else:
                prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred,
                                            "score": node_score, "ctx": retrieval_docs[i], "parent": parent_node,
                                            "overall_score_dict": overall_score_dict}
            # Meet:
            if "[Retrieval]" in curr_pred:
                gen_result_index = curr_pred.index("[Retrieval]")
                prev_generation = curr_pred[:gen_result_index]
            else:
                prev_generation = curr_pred
            '''
            Diff: check wrong pattern and cutting the wrong pattern in curr_pred.
            '''
            prediction_tree[node_id]["processed_pred"] = prev_generation 
            levels[curr_depth].append(node_id)
        return prediction_tree, node_id, levels
    
    def set_predictionTree_NoRetrieval(self, curr_depth, parent_node, node_id, curr_pred, curr_score, curr_prompt, retrieval_docs, prediction_tree, overall_score_dict, level_tmp):
        curr_pred = curr_pred[0]
        node_id += 1
        if self.realtime_retrieval == True:
            prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred, 
                                        "score": curr_score[0], "ctx": retrieval_docs[1], "parent": parent_node,
                                        "overall_score_dict": overall_score_dict} 
        else:
            prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred, 
                                        "score": curr_score[0], "ctx": retrieval_docs[0], "parent": parent_node,
                                        "overall_score_dict": overall_score_dict} 
        # Meet:
        if "[Retrieval]" in curr_pred:
            gen_result_index = curr_pred.index("[Retrieval]")
            prev_generation = curr_pred[:gen_result_index]
        else:
            prev_generation = curr_pred
        prediction_tree[node_id]["processed_pred"] = prev_generation
        level_tmp.append({'curr_depth':curr_depth, 'node_id':node_id})
        return prediction_tree, node_id, level_tmp

    def run_step_generation_batch(self, prompt, current_retrieval_input, pregiven_passages:Optional[list[dict]],
                                  retrieval_tokens=None, relevant_tokens=None, ground_tokens=None,  utility_tokens=None,
                                  w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False) -> tuple[list[str], list[float], dict]:
        if self.realtime_retrieval == True: 
            passages = self.retrieval.search(current_retrieval_input) # 这块必须的设计一下关于
            evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(passage["title"], passage["content"]) for rank, passage in passages.items()] 
        else:
            evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(para["title"], para["text"]) for para in pregiven_passages] 

        preds = self.llm.generate(evidence_augmented_inputs, self.sampling_params)
        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        final_preds = []
        for p_idx, pred in enumerate(preds): 
            pred_text = pred.outputs[0].text
            print(f'output_text"{pred_text}')
            # calculate seq score
            seq_score = self.sequence_score(pred)
            # init dict in each loop
            relevance_score_dict.setdefault(p_idx, {}) 
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # relevance score
            relevance_score, relevance_score_dict = self.relevanceToken_score(pred, relevant_tokens, p_idx, relevance_score_dict)
            # Issupport score
            ground_score, grd_score_dict = self.IssupportToken_score(pred, ground_tokens, p_idx, grd_score_dict)
            # Utility score
            utility_score, ut_score_dict = self.UtilityToken_score(pred, utility_tokens, p_idx, ut_score_dict) 
            '''
            Diff: selfrag_reproduction.py use self.UtilityToken_score() calculate the correct utility_score, which is different from the logic of selfrag_orignal.py
            '''
            if self.use_seqscore is True:
                final_score = seq_score + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score +  w_sup * ground_score + w_use * utility_score
            overall_scores[p_idx] = {"final_score": final_score} 

            if "[No Retrieval]" in pred_text:
                pred_text = self.modify_NoRetrieval_into_Retrieval(pred, retrieval_tokens)
                final_preds.append(pred_text)
            else:
                final_preds.append(pred_text)
        # end of the "for p_idx, pred in enumerate(preds):"
        preds = final_preds
        scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores] 
        if self.realtime_retrieval == True:
            retrieval_docs = passages
        else:
            retrieval_docs = pregiven_passages # pregiven_passages only provide in PopQA, 
        return preds, scores, overall_scores, retrieval_docs
    


    def get_lastTurn_generation(self, parent_node, prediction_tree):
        ''' 
        get previous information from prediction_tree
        '''
        prev_pred = prediction_tree[parent_node]["pred"]
        prev_prompt = prediction_tree[parent_node]["prompt"]
        prev_generation = prediction_tree[parent_node]["processed_pred"]
        prev_generationScore = prediction_tree[parent_node]["score"]
        return prev_pred, prev_prompt, prev_generation, prev_generationScore
        
    def firstToken_retrievalRatio(self, prompt:str, retrieval_tokens:dict[str,int], generation_track:Optional[dict[str,Any]]) -> tuple[float, dict]:
        '''
        calculate the ratio of retrieval base on first token logits
        '''
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=32000, skip_special_tokens = False)
        '''
        Diff: According to self rag's paper, when calculating the ratio, language model only to predict the next token logits.
              Source code max_tokens is often set to 50, 100 or even 300, which greatly wastes computing resources. 
              Raglab optimizes the process of self-rag inference in selfrag_reproduction.py , improves the speed of reasoning and saves a lot of computing resources
        '''
        preds = self.llm.generate([prompt], sampling_params) 
        pred_log_probs = preds[0].outputs[0].logprobs 
        score_dict = {}
        for tok, id in retrieval_tokens.items():
            if id not in pred_log_probs[0]:
                score_dict[tok] = -100
            prob = pred_log_probs[0][id] 
            score_dict[tok] = np.exp(prob)
            '''
            Diff: Raglab selfrag_reproduction.py fix the bug of "score_dict[tok] = float(prob)" and calculate the right ratio.
            Th bug is from self rag source code [https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py#L79]
            '''
        if "short_form" == self.inference_form:
            generation_track["decide_retrieval_mode"] = preds[0].outputs[0].text 
        ratio = score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])  
        return float(ratio), generation_track

    def sequence_score(self,pred) ->float:
        '''
        average prob of generated sentence
        '''
        score = np.exp(pred.outputs[0].cumulative_logprob) / max(len(pred.outputs[0].token_ids), 1)
        return float(score)

    def relevanceToken_score(self, pred, relevant_tokens:dict[str,int], p_idx:int, relevance_score_dict:dict) -> tuple[float, dict]:
        pred_log_probs = pred.outputs[0].logprobs
        for tok, id in relevant_tokens.items(): 
            prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
            relevance_score_dict[p_idx][tok] = np.exp(float(prob))
        # calculate score
        relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (np.sum(list(relevance_score_dict[p_idx].values())))
        return float(relevance_score), relevance_score_dict

    def IssupportToken_score(self, pred, ground_tokens:dict[str,int], p_idx:int, grd_score_dict:dict) -> tuple[float, dict]:
        pred_token_ids = pred.outputs[0].token_ids
        pred_log_probs = pred.outputs[0].logprobs
        groundness_token_appear_indices = []
        # get the position of Issupport token
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(ground_tokens.values()):
                groundness_token_appear_indices.append(tok_idx)
                break
        if len(groundness_token_appear_indices) > 0:
            idx = groundness_token_appear_indices[0]
            for token, token_id in ground_tokens.items():
                prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100 
                grd_score_dict[p_idx][token] = np.exp(float(prob))
        # calculate score
        if len(grd_score_dict[p_idx]) == 3: 
            gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
            ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (grd_score_dict[p_idx]["[Partially supported]"] / gt_sum) # 
        else:
            ground_score = 0.0 # "If the sentence is labeled as [isRel], then [Issup] will not appear later, resulting in a ground score of 0."
        return float(ground_score), grd_score_dict
    
    def UtilityToken_score(self, pred, utility_tokens:dict, p_idx:int, ut_score_dict:dict) -> tuple[float, dict]:
        pred_token_ids = pred.outputs[0].token_ids
        pred_log_probs = pred.outputs[0].logprobs
        utility_token_appear_indices = []
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(utility_tokens.values()):
                utility_token_appear_indices.append(tok_idx)
        if len(utility_token_appear_indices) > 0:
            idx = utility_token_appear_indices[0] # position of ut_token [Utility:1-5]
            for token, token_id in utility_tokens.items(): 
                '''
                diff: Raglab fix the bug which in selfrag orignal code.
                '''
                prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                ut_score_dict[p_idx][token] = np.exp(float(prob))

        if len(ut_score_dict[p_idx]) == 5: 
            ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
            ut_scores = [-1, -0.5, 0, 0.5, 1]
            utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
        else:   
            utility_score = 0.0
        return float(utility_score), ut_score_dict

    def modify_NoRetrieval_into_Retrieval(self,pred, retrieval_tokens)-> str:
        '''
        check the ratio of ([Retrieval] + [Continue to Use Evidence])/([Retrieval] + [Continue to Use Evidence] + [No Retrieval] )
        if the ratio > threshold modify [No Retrieval] -> [Retrieval]
        '''
        pred_text = pred.outputs[0].text
        pred_log_probs = pred.outputs[0].logprobs 
        pred_token_ids = pred.outputs[0].token_ids
        ret_token_appear_indices = []
        substrings = pred_text.split("[No Retrieval]")
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok == retrieval_tokens["[No Retrieval]"]:
                ret_token_appear_indices.append(tok_idx)
                substrings
                print("retrieval_tokens")

        ret_token_score_dict = {}
        retrieval_remap = {}
        for order, idx in enumerate(ret_token_appear_indices):
            ret_token_score_dict.setdefault(order, {})
            for tok, tok_id in retrieval_tokens.items(): 
                prob = pred_log_probs[idx][tok_id] if tok_id in pred_log_probs[idx] else -100
                ret_token_score_dict[order][tok] = np.exp(prob)
            if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0: 
                do_retrieve = (ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[Continue to Use Evidence]"]) / (
                    ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]) > self.threshold
            else:
                do_retrieve = 0.0
            if do_retrieve > self.threshold:
                retrieval_remap[order] = True
            else:
                retrieval_remap[order] = False
        processed_pred = ""
        for substr_i, substring in enumerate(iterable=substrings):
            if substr_i in retrieval_remap and retrieval_remap[substr_i] is True:
                processed_pred += substring + "[Retrieval]" 
            else:
                processed_pred += substring + "[No Retrieval]"
        return processed_pred