from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import numpy as np

from raglab.dataset import PopQA, PubHealth, ArcChallenge, TriviaQA, MultiChoiceQA
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
from raglab.rag.infer_alg.self_rag_reproduction.utils import load_special_tokens, postprocess_answer_option_conditioned, preprocess_input_data
from raglab.rag.infer_alg.self_rag_reproduction.utils import PROMPT_DICT, process_data_evidences

import pudb
from tqdm import tqdm

class SelfRag_Reproduction(NaiveRag):
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
            response, generation_track, do_retrieve = self.short_form_generation(input, source_question,evidences, max_new_tokens = self.generate_maxlength, 
                            use_seqscore = self.use_seqscore, threshold = self.threshold,
                            w_rel = self.w_rel, w_sup = self.w_sup, w_use = self.w_use, 
                            mode = self.retrieval_mode, 
                            show_specialtokens = self.show_specialtokens) 
            return response, generation_track, do_retrieve
        elif 'evaluation' == mode:
            # difine dataset
            if 'PopQA' == self.task:
                self.EvalData = PopQA(self.output_dir, self.llm_path, self.eval_datapath)
            elif 'PubHealth' == self.task:
                self.EvalData = PubHealth(self.output_dir, self.llm_path, self.eval_datapath)
            elif 'ArcChallenge' == self.task:
                self.EvalData = ArcChallenge(self.output_dir, self.llm_path, self.eval_datapath)
            elif 'TriviaQA' == self.task:
                self.EvalData = TriviaQA(self.output_dir, self.llm_path, self.eval_datapath)
            
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
                response, generation_track, do_retrieve = self.short_form_generation(input, source_question, evidences, max_new_tokens = self.generate_maxlength, 
                                                        use_seqscore = self.use_seqscore, threshold = self.threshold,
                                                        w_rel = self.w_rel, w_sup = self.w_sup, w_use = self.w_use, 
                                                        mode = self.retrieval_mode, 
                                                        show_specialtokens = self.show_specialtokens)
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
        
    def short_form_generation(self, prompt, source_question, evidences = None, max_new_tokens = 300,
                    use_seqscore=False, threshold=0.2,
                    w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", show_specialtokens = True): 
        # args init
        #load special token
        
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        self.tokenizer, use_grounding=self.use_groundness, use_utility=self.use_utility)
        results = {}
        # diff retrieval_mode
        if 'always_retrieval' == self.retrieval_mode:
            do_retrieve = True
        elif 'no_retrieval' == self.retrieval_mode:
            do_retrieve = False
        elif 'adaptive_retrieval' == self.retrieval_mode:
            #retrieval or not base on first token
            ratio, results = self.first_token_retrievalRatio(prompt, ret_tokens, results)
            do_retrieve = ratio > threshold
        # "do retrieval or not retrieval
        if do_retrieve is True:             
            #TODO 使用 colbert 的时候还需要添加 title 属性这个相对难一些
            if self.realtime_retrieval == True: # 使用本地的 passages 进行复现
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
        if len(results) <= 2: # 因为我给 no retrieval 也添加了 reult，也就是说 adaptive 当中如果走了 no retrieval 的逻辑，这个时候 result len == 2，所以走了第二个逻辑，
            # post process for no retrieval
            if True == self.show_specialtokens: 
                return pred, results, do_retrieve
            else:
                # remove all sprcial tokens 
                postprocessed_pred = postprocess_answer_option_conditioned(pred) 
                return postprocessed_pred, results, do_retrieve 
        else:
            answer2score = {}
            if isinstance(self.EvalData, MultiChoiceQA): #判断是否是 MultiChoiceQA 如果是则使用另一个 rank 的方法
                '''
                Aggregating for multi-choice question
                source explaination: For SELF-RAG inference on PubHealth and ARC-C, instead of determining the output with the highest score as in other tasks, 
                                    we aggregate the scores for each option and select the answer option with the highest score.       
                paper: https://arxiv.org/abs/2310.11511
                
                '''
                for key, result in results.items(): #TODO debug for this code
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
                path2score = {key: item["score"] for key,
                            item in results.items() if key != "decide_retrieval_mode"} 
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

    def first_token_retrievalRatio(self, prompt, ret_tokens, results):
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
            score_dict[tok] = np.exp(float(prob)) # Diff: diff from the source code [https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py#L79]
        results["decide_retrieval_mode"] = preds[0].outputs[0].text 
        ratio = score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])  
        return float(ratio), results

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
                prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100 # 如果没有那就说明其概率非常的小，超出top-5000 的范围这个时候给一个非常小的概率即可
                grd_score_dict[p_idx][token] = np.exp(float(prob))
        if len(grd_score_dict[p_idx]) == 3: #
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
        # calculate the score of utility token
        if len(ut_score_dict[p_idx]) == 5: 
            ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
            ut_scores = [-1, -0.5, 0, 0.5, 1]
            utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
        else:   
            utility_score = 0.0
        return float(utility_score), ut_score_dict