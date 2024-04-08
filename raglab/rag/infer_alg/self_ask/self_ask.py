from typing import Optional
from tqdm import tqdm
import pdb
import re
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag, ModeNotFoundError
from pprint import pprint
class SelfAsk(NaiveRag):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, query:Optional[str] = None, mode = 'interact'):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            final_answer, generation_track = self.infer(query)
            return final_answer
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset()
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                eval_data = self.EvalData.preprocess(eval_data) # some dataset need preprocess such as: arc_challenge
                question = eval_data[self.EvalData.inputStruction.question]
                output, generation_track = self.infer(question)
                pprint(generation_track)
                inference_results = self.EvalData.record_result(eval_data, output, inference_results)
                pprint(f'output:{output} \n eval_data: {eval_data[self.EvalData.inputStruction.answer]}')
                acc = self.EvalData.eval_acc(inference_results)
                EM = self.EvalData.eval_exact_match(inference_results)
                f1_score = self.EvalData.eval_f1_score(inference_results)
                pprint(f'{self.task} in {idx} turn: \n Accuracy: {acc} \n Exact match:{EM} \n F1 score: {f1_score}')
            # end of for loop
            self.EvalData.save_result(inference_results)
            acc = self.EvalData.eval_acc(inference_results)
            EM = self.EvalData.eval_exact_match(inference_results)
            f1_score = self.EvalData.eval_f1_score(inference_results)
            print(f'{self.task} in {idx} turn: \n Accuracy: {acc} \n Exact match:{EM} \n F1 score: {f1_score}')
            eval_result = {'Accuracy':acc, 'Exact match': EM, 'F1 score':f1_score}
            return eval_result
        else:
            raise ModeNotFoundError("Mode must be interact or evaluation. Please provide a valid mode.")

    def infer(self, query:str) -> tuple[str, dict]:
        '''
        Instruction: our self-ask instructions strictly follow the instructions provided by the original paper and open-source code. 
                     Considering the window length of local llm(llama2-7b ,mistral, etc.), we only used one-shot instead of using the four-shot in the self ask open source code.
        paper:[https://arxiv.org/abs/2210.03350]        
        github code: [https://github.com/ofirpress/self-ask]
        '''
        target_instruction = self.find_instruction('self_ask-followup_question', self.task)
        input_with_followup = target_instruction.format_map({'query': query})
        follow_up = self.llm_inference(input_with_followup)
        generation_track = {}
        turn_idx = 1
        if 'Follow up:' in follow_up:
            while 'Follow up:' in follow_up:
                followup_question = self._extract_followup(follow_up)
                if followup_question == '':
                    print(f'Bad case!!!')
                    break
                passages = self.retrieval.search(followup_question)
                collated_passages = self.collate_passages(passages)
                target_instruction = self.find_instruction('self_ask-read', self.task) 
                input_with_passages = target_instruction.format_map({'passages': collated_passages, 'query': followup_question})
                intermediate_answer = self.llm_inference(input_with_passages) 
                generation_track[turn_idx] = {
                                                'follow up question': followup_question,
                                                'intermediate answer': intermediate_answer,
                                                'cite passages': passages
                                              }
                turn_idx += 1
                first_part = self._extract_first_part(follow_up)
                input_with_followup = input_with_followup + first_part + intermediate_answer + ' \n '
                follow_up = self.llm_inference(input_with_followup)
                if 'So the final answer is:' in follow_up : 
                    follow_up = self._extract_final_answer(follow_up)
                    break
                if follow_up == '':
                    # some special case will generate ''. In this situation we need add instruction for self ask finish the whole inference
                    follow_up = self.llm_inference(input_with_followup + 'So the final answer is:')
                    break
            # end of while
        else:
            passages = self.retrieval.search(query)
            collated_passages = self.collate_passages(passages)
            target_instruction = self.find_instruction('self_ask-read', self.task)
            input = target_instruction.format_map({'passages': collated_passages, 'query': query})
            follow_up = self.llm_inference(input)
            generation_track['cite passages'] = passages
        generation_track['final answer'] = follow_up
        final_answer = follow_up
        return final_answer, generation_track

    def _extract_followup(self, followup):
        followup_pattern = r'Follow up: (.+)\n'
        result = re.findall(followup_pattern, followup)
        followup_question = ''
        if len(result) >= 1:
            followup_question = result[0]
        return followup_question
    
    def _extract_first_part(self, followup):
        followup_pattern = r'^(.+?Intermediate Answer: )'
        result = re.findall(followup_pattern, followup, re.DOTALL) # re.DOTALL flag instructs the regex engine to allow . to match any character, including newline \n.
        followup_question = ''
        if len(result) >= 1:
            followup_question = result[0]
        return followup_question
    
    def _extract_final_answer(self, followup):
        followup_pattern = r'So the final answer is: (.+)'
        result = re.findall(followup_pattern, followup, re.DOTALL) # re.DOTALL flag instructs the regex engine to allow . to match any character, including newline \n.
        followup_question = ''
        if len(result) >= 1:
            followup_question = result[0]
        return followup_question