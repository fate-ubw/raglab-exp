
from typing import Optional, Any
from tqdm import tqdm
import re
import numpy as np 
from raglab.dataset.utils import get_dataset
from raglab.rag.infer_alg.self_rag_reproduction.selfrag_reproduction import SelfRag_Reproduction
from raglab.language_model import VLLM_Client
from raglab.rag.infer_alg.unified_critic_model.utils import load_special_tokens
from raglab.instruction_lab import ALGORITHM_INSTRUCTIONS, DATA_INSTRUCTIONS, SYSTEM_INSTRUCTION
import pdb

class UnifiedCriticRAGFineRank(SelfRag_Reproduction):
    def __init__(self,args):
        super().__init__(args)

    def init(self, args):
        self.dtype = args.dtype
        self.threshold = args.threshold
        self.use_groundness = args.use_groundness
        self.use_utility = args.use_utility
        self.w_rel = args.w_rel
        self.w_sup = args.w_sup
        self.w_use = args.w_use
        # change llm_path -> 
        args.llm_path = args.critic_path
        args.use_vllm = True
        self.critic_model = self.steup_llm(args)

    def inference(self, query:Optional[str]=None, mode='interact', task=None):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            self.print_fn(f"Interactive mode: query = {query}")
            final_answer, generation_track = self.infer(query)
            return final_answer, generation_track
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self)
            
            self.eval_dataset = self.EvalData.load_dataset()
            self.print_fn(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                eval_data = self.EvalData.preprocess(eval_data)
                question = eval_data[self.EvalData.InputStruction.question] 
                outputs, generation_track = self.infer(question)
                inference_results = self.EvalData.record_result(eval_data, outputs, inference_results)
                self.print_fn(f'{self.algorithm_name} {self.task} in {idx+1} turn:\n Question:{question} \n Rag Output:{outputs} \n Answers: {eval_data[self.EvalData.InputStruction.answer]}')
                # calculate metric
                if self.task in ['ASQA', 'Factscore']:
                    continue
                acc = self.EvalData.eval_acc(inference_results)
                EM = self.EvalData.eval_exact_match(inference_results)
                f1_score = self.EvalData.eval_f1_score(inference_results)
                self.print_fn(f'{self.algorithm_name} {self.task} in {idx+1} turn: \n Accuracy: {acc} \n Exact match:{EM} \n F1 score: {f1_score}')
            # --> end of loop
            self.EvalData.save_result(inference_results)
            if self.task in ['ASQA', 'Factscore']:
                return 'Inference completion'
            else:
                eval_result = {'Accuracy':acc, 'Exact match': EM, 'F1 score':f1_score}
                self.EvalData.save_evaluation_results(eval_result)
                return eval_result
        else:
            raise ModeNotFoundError("Mode must be interact or evaluation. Please provide a valid mode.")

    def infer(self, query):
        generation_track = {}
        branch_track = {}
        retrieval_tokens, relevant_tokens, ground_tokens, utility_tokens = load_special_tokens(self.critic_model.tokenizer, use_grounding = self.use_groundness, use_utility = self.use_utility)
        # always retrieval: no need for [Retrieval]
        passages = self.retrieval.search(query) # retrieval input is query
        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        for p_idx, passage in passages.items():
            formatted_passage = passage['title'] + '\n' + passage['text'] # aligned with instruction
            relevance_score_dict.setdefault(p_idx, {}) 
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # [relevance]
            system_instruction = SYSTEM_INSTRUCTION['normal_prompt_input']
            relevance_instruction = self.find_algorithm_instruction('critic-relevance_instruction', None)
            relevance_input = self.find_algorithm_instruction('critic-relevance_input', None)
            formatted_relevance_input = relevance_input.format_map({'instruction':query, "evidence":formatted_passage})
            input = system_instruction.format_map({'instruction':relevance_instruction, 'input':formatted_relevance_input})
            outputlist = self.critic_model.generate(input)
            relevance_score, relevance_score_dict = self._relevanceToken_score(outputlist[0], relevant_tokens, p_idx, relevance_score_dict)
            generation_track['branch_{}-critic_relevance_result'.format(p_idx)] = outputlist[0].text
            # base model 
            
            system_instruction_no_input = SYSTEM_INSTRUCTION['normal_prompt_no_input']
            basemodel_instruction = self.find_algorithm_instruction('base_model_instruction',None)
            task_instruction = self.find_dataset_instruction(self.task)
            formatted_basemodel_instruction = basemodel_instruction.format_map({'task_instruction':task_instruction,'query':query, 'passages': formatted_passage}) 
            input = system_instruction_no_input.format_map({'instruction':formatted_basemodel_instruction})
            outputlist = self.llm.generate(input)
            base_model_output = outputlist[0].text
            generation_track['branch_{}-base_model_output'.format(p_idx)] = base_model_output
            # [IsSup]
            question_with_task_instruct = f'{task_instruction} {query}'
            groundness_instruction = self.find_algorithm_instruction('critic-ground_instruction', None)
            groundness_input = self.find_algorithm_instruction('critic-ground_input',None)
            formatted_groundness_input = groundness_input.format_map({'instruction':question_with_task_instruct, 'evidence':formatted_passage, 'output':base_model_output})
            input = system_instruction.format_map({'instruction':groundness_instruction, 'input':formatted_groundness_input})
            outputlist = self.critic_model.generate(input)
            ground_score, grd_score_dict = self._IssupportToken_score(outputlist[0], ground_tokens, p_idx, grd_score_dict)
            generation_track['branch_{}-critic_is_support_result'.format(p_idx)] = outputlist[0].text
            #[utility]
            utility_instruction = self.find_algorithm_instruction('critic-utility_instruction',None)
            utility_input = self.find_algorithm_instruction('critic-utility_input',None)
            formatted_utility_input = utility_input.format_map({'instruction':question_with_task_instruct, 'output':base_model_output})
            input = system_instruction.format_map({'instruction':utility_instruction,'input':formatted_utility_input})
            outputlist = self.critic_model.generate(input)
            utility_score, ut_score_dict = self._UtilityToken_score(outputlist[0], utility_tokens, p_idx, ut_score_dict)
            generation_track['branch_{}-critic_utility_result'.format(p_idx)] = outputlist[0].text
            final_score = self.w_rel * relevance_score + self.w_sup * ground_score + self.w_use * utility_score
            branch_track['branch_{}'.format(p_idx)] = {'pred':base_model_output, 'score': float(final_score), 'passage': passage}
        # --> end of passage loop
        # rank and get top-1 branch
        path2score = {key: item['score'] for key, item in branch_track.items()}
        ranked_path = sorted(path2score.items(), key=lambda x:x[1], reverse=True)
        top_1_path = ranked_path[0][0]
        top_2_path = ranked_path[1][0]
        
        top_1_response = branch_track[top_1_path]['pred']
        top_2_response = branch_track[top_2_path]['pred']

        # pairwise improvement_answer & top_response
        pair_wise_instruction = self.find_algorithm_instruction('critic-pair_wise-instruction', None)
        formatted_pair_wise_instruction = pair_wise_instruction.format_map({'instruction':question_with_task_instruct, 'response_1':top_1_response, 'response_2':top_2_response})
        input_1 = system_instruction_no_input.format_map({'instruction':formatted_pair_wise_instruction})
        outputlist = self.critic_model.generate(input_1)
        output_1 = outputlist[0].text
        result_1 = self.extract_pairwise_result(output_1)
        generation_track['pair-wise_turn_1_result'] = output_1
        formatted_pair_wise_instruction = pair_wise_instruction.format_map({'instruction':question_with_task_instruct, 'response_1':top_2_response, 'response_2':top_1_response})
        input_2 = system_instruction_no_input.format_map({'instruction':formatted_pair_wise_instruction})
        outputlist = self.critic_model.generate(input_2)
        output_2 = outputlist[0].text
        result_2 = self.extract_pairwise_result(output_2)
        generation_track['pair-wise_turn_2_result'] = output_2

        if result_1 == '1' and result_2 == '2':
            final_answer = top_1_response
        elif result_1 == '2' and result_2 == '1':
            final_answer = top_2_response
        elif result_1 == result_2: # tie 
            final_answer = top_1_response # 这里面到底使用什么比较合适呢？
        else:
            final_answer = top_1_response
        return final_answer, generation_track

    def find_dataset_instruction(self, dataset_name:str) -> str:
        target_instruction = ''
        for instruction in DATA_INSTRUCTIONS:
            if instruction["dataset_name"].lower() == dataset_name.lower():
                target_instruction = instruction["instruction"]
        return target_instruction

    def _UtilityToken_score(self, pred: VLLM_Client.Outputs, utility_tokens:dict, p_idx:int, ut_score_dict:dict) -> tuple[float, dict]:
        pred_token_ids = pred.tokens_ids
        pred_log_probs = pred.logprobs # list[dict['volcab id',logprobs]]
        utility_token_appear_indices = []
        for tok_idx, tok in enumerate(pred_token_ids): 
            if tok in list(utility_tokens.values()):
                utility_token_appear_indices.append(tok_idx)
        if len(utility_token_appear_indices) > 0: 
            idx = utility_token_appear_indices[0] # position of ut_token [Utility:1-5]
            for token, token_id in utility_tokens.items(): 
                prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                ut_score_dict[p_idx][token] = np.exp(float(prob))

        if len(ut_score_dict[p_idx]) == 2: 
            ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
            utility_score = ut_score_dict[p_idx]['[Utility:2]'] / ut_sum
        else:   
            utility_score = 0.0
        return float(utility_score), ut_score_dict

    def extract_pairwise_result(self, output:str):
        pairwise_result = output.split()[2]
        if pairwise_result not in ['1', '2', 'tie']:
            pairwise_result = None
        return pairwise_result


class ModeNotFoundError(Exception):
    pass