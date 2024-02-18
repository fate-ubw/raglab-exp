from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import numpy as np

from raglab.dataset.utils import get_dataset
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
from raglab.rag.infer_alg.self_rag_original.utils import load_special_tokens, postprocess_answer_option_conditioned, preprocess_input_data
from raglab.rag.infer_alg.self_rag_original.utils import PROMPT_DICT, process_data_evidences

import pudb
from tqdm import tqdm

class SelfRag_Original(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        self.init(args)

    def init(self, args):
        self.download_dir = args.download_dir
        self.world_size = args.world_size
        self.dtype = args.dtype
        self.threshold = args.threshold
        self.use_seqscore = args.use_seqscore
        self.use_groundness = args.use_groundness
        self.use_utility = args.use_utility
        self.beam_width = args.beam_width
        self.max_depth = args.max_depth
        self.w_rel = args.w_rel
        self.w_sup = args.w_sup
        self.w_use = args.w_use
        self.retrieval_mode = args.retrieval_mode
        self.show_specialtokens = args.show_specialtokens
        self.realtime_retrieval = args.realtime_retrieval

    def inference(self, query=None, mode='interact', task=None):
        assert mode in ['interact', 'evaluation'] 
        if 'interact' == mode:
            input = f"### Instruction:\n{query}\n\n### Response:\n" # add inference format 
            source_question = query
            response, generation_track, do_retrieve = self.short_form_generation(input, source_question, evidences,
                                                    use_seqscore = self.use_seqscore, threshold = self.threshold,
                                                    w_rel = self.w_rel, w_sup = self.w_sup, w_use = self.w_use )
            return response, generation_track, do_retrieve
        elif 'evaluation' == mode:
            # get dataset
            self.EvalData = get_dataset(self.task, self.output_dir,self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset()
            #TODO  preprocess data 应该放在每一个 dataset class 方法下面,和具体数据绑定
            self.eval_dataset = preprocess_input_data(self.eval_dataset, task = self.task) # find task instruction 
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                temp = {}
                source_question = eval_data['question'] 
                input = PROMPT_DICT["prompt_no_input"].format_map(eval_data) # get instruction 

                if self.realtime_retrieval == True:
                    evidences = []
                else:
                    _, evidences = process_data_evidences(eval_data, self.n_docs) # use pre-given passages and do not use retrieval model in real-time
                response, generation_track, do_retrieve = self.short_form_generation(input, source_question, evidences,
                                                        use_seqscore = self.use_seqscore, threshold = self.threshold,
                                                        w_rel = self.w_rel, w_sup = self.w_sup, w_use = self.w_use )
                print(f'source question:{source_question}')
                print(f'response: {response}')
                if "SUPPORTS" in response: # the trick in self rag source code. In some situation LLM will generate SUPPORTS or REFUTES instead of true or flase
                    response = "true" 
                elif "REFUTES" in response: 
                    response = "false"

                temp['question'] = source_question
                temp['answers'] = eval_data['answers']
                temp['generation'] = response
                temp['instruction'] = input
                temp['generation_track'] = generation_track
                inference_results.append(temp)
                # calculate the error in each step
                eval_result = self.EvalData.eval_acc(inference_results)
                print(f'{self.task} Accuracy in {idx} turn: {eval_result}')
            # end of for loop
            self.EvalData.save_result(inference_results)
            eval_result = self.EvalData.eval_acc(inference_results)
            print(f'Final {self.task} accuracy: {eval_result}')
            return eval_result

    def load_llm(self): 
        llm = LLM(model=self.llm_path)
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_path, padding_side="left")
        return llm, tokenizer, sampling_params

    def get_instruction(self, query):
        instruction = query
        return instruction
        

    def long_form_generation(prompt, query, evidence=None, max_new_tokens=300,                               
                                     beam_width=2, max_depth=7,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, ignore_cont = None): # orignal version of self rag longform
        if "## Input:\n\n" in query:
            query = query.split("## Input:\n\n")[1]
        #prompt 放到外面去管理
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
                                self.tokenizer, use_grounding=self.use_grounding, use_utility=self.use_utility)
        special_tokens = [] 

        if rel_tokens is not None:
            special_tokens = list(rel_tokens.keys())
        if ret_tokens is not None:
            special_tokens += list(ret_tokens.keys())

        if  "no_retrieval" == self.mode: 
            prompt += "[No Retrieval]" 
            preds = self.llm.generate([prompt], self.sampling_params)
            preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
            return preds[0]

        elif "always_retrieve" == self.mode:
            do_retrieve = True
        elif 'adaptive_retrieval' == self.mode: 
            do_retrieve = self.firstToken_retrievalRatio_longForm(prompt, ret_tokens)

        if do_retrieve is False:
            # no retrieval
            prompt += "[No Retrieval]"
            preds = self.llm.generate([prompt], self.sampling_params)
            preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
            prediction_tree = {}
            return preds[0], prediction_tree # 也就是说这个逻辑还有一个问题，如果这一次是【No retrieval】那么就直接返回了，后面就不会进行 retrieval。这个逻辑还是有问题的，因为
            # 代码到这里就直接返回了，所以代码肯定出 bug 啊这里感觉就不能使用[Retrieval] 这个special tokens 真的不知道它是怎么评测的 
        elif do_retrieve is True:
            # 开始 always or adaptive retrieval
            curr_depth = 1 
            terminated = False 
            node_id = 0 
            prediction_tree = {} 
            levels = {} 
            prediction_tree[node_id] = {"prompt": prompt, "pred": "[Retrieval]", 
                                        "processed_pred": "", "score": None, "ctx": None, "parent": None}
            levels[0] = [0]
            while curr_depth < max_depth: 
                # 构建整个树
                levels[curr_depth] = []
                if curr_depth-1 in levels and terminated is False:
                    for parent_node in levels[curr_depth-1]:
                        # 这里可以给一个函数，get_lastTurn_generation(parent_node, prediction_tree):
                        prev_pred, prompt, prev_generation, prev_score = self.get_lastTurn_generation(parent_node, prediction_tree)
                        if prev_pred == "</s>":
                            terminated = True
                            continue
                        if "[Retrieval]" in prev_pred:
                            curr_prompt = prompt + prev_generation # get new prompt
                            curr_preds, curr_scores, overall_score_dict = run_step_generation_batch( 
                                                                        model, curr_prompt, ctxs, max_new_tokens,
                                                                        rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                                        threshold=threshold, w_rel=w_rel, w_sup=w_sup, w_use=w_use)
                            # set_prediction_tree():
                            prediction_tree, node_id = self.set_predictionTree(curr_depth, parent_node, node_id, curr_preds, curr_scores, curr_prompt, 
                                                                            prev_score, ctxs, prediction_tree, levels ,overall_score_dict)
                    current_rank = levels[curr_depth]# 当curr_depth = 2 时候取出来的current_rank是空的，
                    node2score = { node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                    top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[:beam_width] # 取 top2 结果
                    levels[curr_depth] = [node[0] for node in top_nodes] 
                    curr_depth += 1  
                else:
                    break

        best_selections = self.backtracking_prediction_tree(levels, curr_depth, prediction_tree)
        # get final_prediction 
        final_prediction = {}
        splitted_sentences = {}
        original_splitted_sentences = {}
        ctxs = {}
        for path_i, nodes in best_selections.items(): # 
            # (Pdb) nodes = [None, 0, 5] 
            final_prediction[path_i] = " ".join([prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))])
            splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
            original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

            ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
                ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

        result = {"final_prediction": final_prediction,
                "splitted_sentences": splitted_sentences,
                "original_splitted_sentences": original_splitted_sentences,
                "best_selections": best_selections,
                "ctxs": ctxs,
                "prediction_tree": prediction_tree}
    
        return final_prediction, result

    def backtracking_prediction_tree(self, levels: dict[int,list[int]], curr_depth: int, prediction_tree: dict[int, dict]) -> dict[int,list[int]]:
        '''
        get best tracking from prediction_tree base on levels
        '''
        parent = 0 
        best_selections = {}
        # Traverse from the bottom 
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0} # remove empty list in levels
        for path_i, node in enumerate(levels[len(levels)]):
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
                # (Pdb) best_selections = {0: [0, 5]}
                current_node = parent 
                current_level += 1
        return best_selections
    
    def set_predictionTree(self, curr_depth, parent_node, node_id,  curr_preds, curr_scores, curr_prompt, prev_score, ctxs, prediction_tree, levels , overall_score_dict):
        retrieval_results = {}
        for i, (curr_pred, p_score) in enumerate(zip(curr_preds, curr_scores)):
            retrieval_results[i] = {"pred": curr_pred, "score": p_score}

        for i, result in retrieval_results.items(): 
            node_id += 1 
            node_score = result["score"] * prev_score if prev_score is not None else result["score"]
            curr_pred = result["pred"] 
            prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred, 
                                        "score": node_score, "ctx": ctxs[i], "parent": parent_node,
                                        "overall_score_dict": overall_score_dict} 
            
            if "[Retrieval]" in curr_pred: 
                gen_result_index = curr_pred.index("[Retrieval]") 
                prev_generation = curr_pred[:gen_result_index] 
            else: 
                prev_generation = curr_pred
            
            prediction_tree[node_id]["processed_pred"] = prev_generation 
            levels[curr_depth].append(node_id)
        return prediction_tree, node_id

    def get_lastTurn_generation(parent_node, prediction_tree):
        # get previous information
        prev_pred = prediction_tree[parent_node]["pred"]
        prev_prompt = prediction_tree[parent_node]["prompt"]
        prev_generation = prediction_tree[parent_node]["processed_pred"]
        prev_generationScore = prediction_tree[parent_node]["score"]
        return prev_pred, prev_prompt, prev_generation, prev_generationScore

    def short_form_generation(self, prompt, source_question, evidences = None,
                            use_seqscore=False, threshold=0.2,w_rel=1.0, w_sup=1.0, w_use=0.5): 
        
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
                                self.tokenizer, use_grounding=self.use_groundness, use_utility=self.use_utility)
        results = {}
        if 'always_retrieval' == self.retrieval_mode:
            do_retrieve = True
        elif 'no_retrieval' == self.retrieval_mode:
            do_retrieve = False
        elif 'adaptive_retrieval' == self.retrieval_mode:
            #retrieval or not base on first token
            ratio, results = self.firstToken_retrievalRatio_shortForm(prompt, ret_tokens, results)
            do_retrieve = ratio > threshold
        # "do retrieval or not retrieval
        if do_retrieve is True:             
            if self.realtime_retrieval == True:
                passages = self.retrieval.search(source_question) # attention: you need source question as retrieval input
                evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
                passage["title"], passage["content"]) for rank, passage in passages.items()]
            else:
                evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(para["title"], para["text"]) for para in evidences] 
            
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
                relevance_score, relevance_score_dict = self.relevanceToken_score(pred, rel_tokens, p_idx, relevance_score_dict)
                # Issupport score
                ground_score, grd_score_dict = self.IssupportToken_score(pred, grd_tokens, p_idx, grd_score_dict)
                # Utility score
                utility_score, ut_score_dict = self.UtilityToken_score(pred, ut_tokens, p_idx, ut_score_dict)
                
                if use_seqscore is True:
                    final_score = np.exp(seq_score) + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score # 涉及不同类型数据转化的一定要涉及类型的转换和精度问题
                else:
                    final_score = w_rel * relevance_score +  w_sup * ground_score + w_use * utility_score
                overall_scores[p_idx] = {"final_score": final_score,
                                        "relevance_score": relevance_score,
                                        "ground_score": ground_score,
                                        "utility_score": utility_score,
                                        "relevance_score_dict": relevance_score_dict, 
                                        "grd_score_dict": grd_score_dict,
                                        "ut_score_dict": ut_score_dict} 
                pred_text = pred.outputs[0].text 
                if self.realtime_retrieval == True:
                    results["retrieval_{}".format(p_idx)] = {"pred": pred_text, "score": float(final_score), "ctx": passages[p_idx+1]} 
                else:
                    results["retrieval_{}".format(p_idx)] = {"pred": pred_text, "score": float(final_score), "ctx": evidences[p_idx]}
        else: 
            # no retrieval generation
            prompt += "[No Retrieval]"
            preds = self.llm.generate([prompt], self.sampling_params)
            pred = preds[0].outputs[0].text 
            results['no_retrieval'] = {"pred": pred} # no retrieval no need score and passages
        
        # Aggregating answers
        if len(results) <= 2: 
            # post process for no retrieval
            if True == self.show_specialtokens: 
                return pred, results, do_retrieve
            else:
                # remove all sprcial tokens 
                postprocessed_pred = postprocess_answer_option_conditioned(pred) 
                return postprocessed_pred, results, do_retrieve 
        else:
            answer2score = {}
            if isinstance(self.EvalData, MultiChoiceQA): 
                '''
                Aggregating for multi-choice question
                source explaination: For SELF-RAG inference on PubHealth and ARC-C, instead of determining the output with the highest score as in other tasks, 
                                    we aggregate the scores for each option and select the answer option with the highest score.       
                paper: https://arxiv.org/abs/2310.11511
                
                '''
                for key, result in results.items():
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
                path2score = {key: item["score"] for key, item in results.items() if key != "decide_retrieval_mode"} 
                # (Pdb) path2score {'retrieval_0': 3.4123800585196546, 'retrieval_1': 2.27039496913239, 'retrieval_2': 3.4020720076164856, 'retrieval_3': 2.6283043364201686, 'retrieval_4': 3.722096903736915, 'retrieval_5': 3.461728838250881, 'retrieval_6': 1.6601180656216912, 'retrieval_7': 2.9027644863792044, 'retrieval_8': 2.852774340193746, 'retrieval_9': 2.2013860727179604}
                best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][0]
                # (Pdb)  best_path: 'retrieval_4'
                best_option = results[best_path]["pred"]
                if self.show_specialtokens == True:
                    pass
                else:
                    # remove all special token 
                    best_option = postprocess_answer_option_conditioned(best_option)

        return best_option, results, do_retrieve 

    def firstToken_retrievalRatio_shortForm(self, prompt, ret_tokens, results):
        '''
        calculate the ratio of retrieval base on first token
        '''
        preds = self.llm.generate([prompt], self.sampling_params)
        pred_log_probs = preds[0].outputs[0].logprobs 
        score_dict = {}
        for tok, id in ret_tokens.items():
            if id not in pred_log_probs[0]: #[0] get the first token
                score_dict[tok] = -100
            prob = pred_log_probs[0][id] # get the special logprob
            score_dict[tok] = float(prob) # Diff:
            # TODO this code should be: score_dict[tok] = np.exp(float(prob)) 
            # This bug is from self rag source code [https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py#L79]
            # The correct version of self rag referenced in Raglab's Selfrag-correct 
        results["decide_retrieval_mode"] = preds[0].outputs[0].text 
        ratio = score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])  
        return float(ratio), results

    def firstToken_retrievalRatio_longForm(self, prompt, ret_tokens):
        # the logic is reference from origanl code
        preds = self.llm.generate([prompt], self.sampling_params)
        pred_log_probs = preds[0].outputs[0].logprobs 
        preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        if "[Retrieval]" not in preds[0]:
            do_retrieve = False
        else:
            if self.threshold is None:
                do_retrieve = False
            else:
                ret_token_score_dict = {}
                for tok, tok_id in ret_tokens.items():
                    prob = pred_log_probs[0][tok_id] 
                    ret_token_score_dict[tok] = np.exp(prob)

                retrieve_prob = ret_token_score_dict["[Retrieval]"] / (ret_token_score_dict["[Retrieval]"] + ret_token_score_dict["[No Retrieval]"])
                do_retrieve = True if retrieve_prob > self.threshold else False
        return  do_retrieve

    def sequence_score(self,pred):
        '''
        average prob of generated sentence
        '''
        score = pred.outputs[0].cumulative_logprob / max(len(pred.outputs[0].token_ids), 1)
        return score

    def relevanceToken_score(self, pred, rel_tokens, p_idx, relevance_score_dict):
        pred_log_probs = pred.outputs[0].logprobs
        for tok, id in rel_tokens.items(): 
            prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100 # 首先判断{'[Irrelevant]': 32003, '[Relevant]': 32004}是否在 pred 里面，如果在里面就取其对应的 logprob，并且是直接取的 ids
            relevance_score_dict[p_idx][tok] = np.exp(float(prob))
        # calculate score
        relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (np.sum(list(relevance_score_dict[p_idx].values())))
        return float(relevance_score), relevance_score_dict

    def IssupportToken_score(self, pred, grd_tokens, p_idx, grd_score_dict):
        pred_token_ids = pred.outputs[0].token_ids
        pred_log_probs = pred.outputs[0].logprobs
        groundness_token_appear_indices = []
        # get the position of Issupport token
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(grd_tokens.values()):
                groundness_token_appear_indices.append(tok_idx)
                break
        if len(groundness_token_appear_indices) > 0:
            idx = groundness_token_appear_indices[0]
            for token, token_id in grd_tokens.items():
                prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100 
                grd_score_dict[p_idx][token] = np.exp(float(prob))
        # calculate score
        if len(grd_score_dict[p_idx]) == 3: 
            gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
            ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (grd_score_dict[p_idx]["[Partially supported]"] / gt_sum) # 
        else:
            ground_score = 0.0 # "If the sentence is labeled as [isRel], then [Issup] will not appear later, resulting in a ground score of 0."
        return float(ground_score), grd_score_dict
    
    def UtilityToken_score(self, pred, ut_tokens, p_idx, ut_score_dict):
        pred_token_ids = pred.outputs[0].token_ids
        pred_log_probs = pred.outputs[0].logprobs
        utility_token_appear_indices = []
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(ut_tokens.values()):
                utility_token_appear_indices.append(tok_idx)
        if len(utility_token_appear_indices) > 0:
            idx = utility_token_appear_indices[0] # position of ut_token [Utility:1-5]
            for token, token_id in ut_tokens.items():
                prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                ut_score_dict[p_idx][token] = np.exp(float(prob))

        if len(ut_score_dict[p_idx]) == 5: 
            ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
            ut_scores = [-1, -0.5, 0, 0.5, 1]
            utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
        else:   
            utility_score = 0.0
        return float(utility_score), ut_score_dict
    
