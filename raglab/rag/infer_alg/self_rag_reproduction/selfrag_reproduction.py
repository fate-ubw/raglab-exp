
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
        self.dtype = args.dtype #

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

    def inference(self, query: Optional[str], mode='interact', task=None):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            input = f"### Instruction:\n{query}\n\n### Response:\n"
            source_question = query
            evidences = []
            response, generation_track, do_retrieve = self.short_form_generation(input, source_question, evidences,
                                                        use_seqscore = self.use_seqscore, threshold = self.threshold,
                                                        w_rel = self.w_rel, w_sup = self.w_sup, w_use = self.w_use, mode = mode)
            return  response, generation_track
        
    def load_llm(self):
        llm = LLM(model=self.llm_path, dtype=self.dtype)
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_path, padding_side="left")
        return llm, tokenizer, sampling_params


    def short_form_generation(self, prompt:str, source_question:str, evidences:Optional[None],
                            use_seqscore=True, threshold=0.2,w_rel=1.0, w_sup=1.0, w_use=0.5, mode = 'evaluation'): 

        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
                                self.tokenizer, use_grounding=self.use_groundness, use_utility=self.use_utility)
        generation_track = {}
        pdb.set_trace()
        if 'always_retrieval' == self.retrieval_mode:
            do_retrieve = True
        elif 'no_retrieval' == self.retrieval_mode:
            do_retrieve = False
        elif 'adaptive_retrieval' == self.retrieval_mode:
            #retrieval or not base on first token
            ratio, generation_track = self.firstToken_retrievalRatio(prompt, ret_tokens, generation_track)
            do_retrieve = ratio > threshold
        # "do retrieval or not retrieval
        if do_retrieve is True:   
            if self.realtime_retrieval == True:
                passages = self.retrieval.search(source_question)
                evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(passage["title"], passage["content"]) for rank, passage in passages.items()] 
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
                    final_score = seq_score + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
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
                    generation_track["retrieval_{}".format(p_idx+1)] = {"pred": pred_text, "score": float(final_score), "ctx": passages[p_idx+1]}
                else:
                    generation_track["retrieval_{}".format(p_idx+1)] = {"pred": pred_text, "score": float(final_score), "ctx": evidences[p_idx]}
        else: 
            # no retrieval generation
            prompt += "[No Retrieval]"
            preds = self.llm.generate([prompt], self.sampling_params)
            pred = preds[0].outputs[0].text 
            generation_track['no_retrieval'] = {"pred": pred} # no retrieval no need score and passages
        
        # Aggregating answers
        if len(generation_track) <= 2: 
            # post process for no retrieval
            if True == self.show_specialtokens: # specialtokens 其实就是很普通的
                return pred, generation_track, do_retrieve
            else:
                # remove all sprcial tokens 
                postprocessed_pred = postprocess_answer_option_conditioned(pred) 
                return postprocessed_pred, generation_track, do_retrieve 
        else:
            answer2score = {}
            pdb.set_trace()
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
                    answer = postprocess_answer_option_conditioned(result["pred"]) # keyword
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
        return best_option, generation_track, do_retrieve 

    def firstToken_retrievalRatio(self, prompt:str, ret_tokens:dict[str,int], generation_track:dict[str,Any]) -> tuple[float, dict]:
        '''
        calculate the ratio of retrieval base on first token logits
        '''
        preds = self.llm.generate([prompt], self.sampling_params)
        pred_log_probs = preds[0].outputs[0].logprobs 
        score_dict = {}
        for tok, id in ret_tokens.items():
            if id not in pred_log_probs[0]:
                score_dict[tok] = -100
            prob = pred_log_probs[0][id] 
            score_dict[tok] = np.exp(prob)
            '''
            Diff: Raglab selfrag_reproduction.py fix the bug of "score_dict[tok] = float(prob)" and calculate the right ratio
            This bug is from self rag source code [https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py#L79]
            '''
        generation_track["decide_retrieval_mode"] = preds[0].outputs[0].text 
        ratio = score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])  
        return float(ratio), generation_track

    def sequence_score(self,pred) ->float:
        '''
        average prob of generated sentence
        '''
        score = np.exp(pred.outputs[0].cumulative_logprob) / max(len(pred.outputs[0].token_ids), 1)
        return float(score)

    def relevanceToken_score(self, pred, rel_tokens:dict[str,int], p_idx:int, relevance_score_dict:dict) -> tuple[float, dict]:
        pred_log_probs = pred.outputs[0].logprobs
        for tok, id in rel_tokens.items(): 
            prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
            relevance_score_dict[p_idx][tok] = np.exp(float(prob))
        # calculate score
        relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (np.sum(list(relevance_score_dict[p_idx].values())))
        return float(relevance_score), relevance_score_dict

    def IssupportToken_score(self, pred, grd_tokens:dict[str,int], p_idx:int, grd_score_dict:dict) -> tuple[float, dict]:
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
    
    def UtilityToken_score(self, pred, ut_tokens:dict, p_idx:int, ut_score_dict:dict) -> tuple[float, dict]:
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