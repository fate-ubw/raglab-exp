from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
from raglab.rag.infer_alg.self_rag.utils import load_special_tokens, postprocess_answer_option_conditioned

class SelfRag(NaiveRag):
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

    def inference(self, query=None, mode='interact', task=None):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            pass   
        elif 'evaluation' == mode:
            if 'PopQA' == self.task:
                pass

    def load_llm(self): 
        llm = LLM(model=self.llm_path)
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_path, padding_side="left")
        return llm, tokenizer

    def get_prompt(self, passages, query):
        return super().get_prompt(passages, query)

    def generation(self, prompt, evidences, max_new_tokens = 300,
                    ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                    use_seqscore=False, threshold=0.2,
                    w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):
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
            ratio, results = first_token_retrievalRatio(prompt, ret_tokens, results)
            do_retrieve = ratio > threshold
        # "do retrieval or not retrieval
        if do_retrieve is True:
            # retrieval.search()
            passages = self.retrieval.search(prompt) 
      
            #TODO 使用 colbert 的时候还需要添加 title 属性这个相对难一些
            evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
            passage["title"], passage["content"]) for rank, passage in passages.items()]
            # evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(para["title"], para["text"]) for para in evidences] 
            
            preds = self.llm.generate(evidence_augmented_inputs, self.sampling_params)
            # calculate score of each candidate
            relevance_score_dict = {}   
            grd_score_dict = {}
            ut_score_dict = {}
            overall_scores = {}
            for p_idx, pred in enumerate(preds): 
                #sequence score 
                seq_score = sequence_score(pred)
                # init dict in each loop
                relevance_score_dict.setdefault(p_idx, {}) 
                grd_score_dict.setdefault(p_idx, {})
                ut_score_dict.setdefault(p_idx, {})
                # relevance score 
                relevance_score, relevance_score_dict = relevanceToken_score(pred, rel_tokens, p_idx, relevance_score_dict)
                # Issupport score
                ground_score, grd_score_dict = IssupportToken_score(pred, grd_tokens, p_idx, grd_score_dict)
                # Utility score
                utility_score, ut_score_dict = UtilityToken_score(pred, ut_tokens, p_idx, ut_score_dict)
                
                if use_seqscore is True:
                    final_score = np.exp(seq_score) + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
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
                results["retrieval_{}".format(p_idx)] = {
                    "pred": pred_text, "score": final_score, "ctx": passages[p_idx+1]} 
        else: # no retrieval generation 
            prompt += "[No Retrieval]"
            preds = self.llm.generate([prompt], self.sampling_params)
            pred = preds[0].outputs[0].text


        # Aggregating answers
        if len(results) == 1: # post process for no retrieval
            postprocessed_pred = postprocess_answer_option_conditioned(pred)
            return postprocessed_pred, results, do_retrieve 
        else:
            answer2score = {}
            if closed is True:
                for key, result in results.items(): #TODO debug for this code
                    if key == "no_retrieval":
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
                            item in results.items() if key != "no_retrieval"} 
                # (Pdb) path2score {'retrieval_0': 3.4123800585196546, 'retrieval_1': 2.27039496913239, 'retrieval_2': 3.4020720076164856, 'retrieval_3': 2.6283043364201686, 'retrieval_4': 3.722096903736915, 'retrieval_5': 3.461728838250881, 'retrieval_6': 1.6601180656216912, 'retrieval_7': 2.9027644863792044, 'retrieval_8': 2.852774340193746, 'retrieval_9': 2.2013860727179604}
                best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][0]
                # best_path: 'retrieval_4'
                best_option = results[best_path]["pred"]
            return best_option, results, do_retrieve 


    def first_token_retrievalRatio(self, prompt, ret_tokens, results):
        '''
        calculate the ratio of retrieval base on first token
        '''
        preds = self.llm.generate([prompt], self.sampling_params)
        pred_log_probs = preds[0].outputs[0].logprobs 
        score_dict = {}
        for tok, id in ret_tokens.items():
            if id not in pred_log_probs[0]: #【0】is first token
                score_dict[tok] = -100
            prob = pred_log_probs[0][id] # get the special logprob
            score_dict[tok] = np.exp(float(prob)) 
        results["no_retrieval"] = preds[0].outputs[0].text 
        ratio = score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])  
        return ratio

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
        return score, relevance_score_dict
    
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
        return ground_score, grd_score_dict
    
    def UtilityToken_score(pred, ut_tokens, p_idx, ut_score_dict):
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
        return utility_score, ut_score_dict