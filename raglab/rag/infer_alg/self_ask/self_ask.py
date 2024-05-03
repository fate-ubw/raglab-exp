import pdb
import re
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
class SelfAsk(NaiveRag):
    def __init__(self, args):
        super().__init__(args)

    def infer(self, query:str) -> tuple[str, dict]:
        '''
        Instruction: our self-ask instructions strictly follow the instructions provided by the original paper and open-source code. 
                     Considering the window length of local llm(llama2-7b ,mistral, etc.), we only used one-shot instead of using the four-shot in the self ask open source code.
        paper:[https://arxiv.org/abs/2210.03350]        
        github code: [https://github.com/ofirpress/self-ask]
        '''
        target_instruction = self.find_instruction('self_ask-followup_question', self.task)
        input_with_followup = target_instruction.format_map({'query': query})
        output_list = self.llm.generate(input_with_followup)
        Output = output_list[0]
        follow_up = Output.text

        print(f'follow up question:{follow_up}')
        generation_track = {}
        turn_idx = 1
        if 'Follow up:' in follow_up:
            while 'Follow up:' in follow_up: 
                followup_question = self._extract_followup(follow_up)
                if followup_question == '':
                    print(f'Bad case!!!')
                    break
                passages = self.retrieval.search(followup_question)
                passages = self._truncate_passages(passages)
                collated_passages = self.collate_passages(passages)
                target_instruction = self.find_instruction('self_ask-read', self.task) 
                input_with_passages = target_instruction.format_map({'passages': collated_passages, 'query': followup_question})
                output_list = self.llm.generate(input_with_passages)
                Output = output_list[0]
                intermediate_answer = Output.text
                generation_track[turn_idx] = {
                                                'follow up question': followup_question,
                                                'intermediate answer': intermediate_answer,
                                                'cite passages': passages
                                              }
                turn_idx += 1
                input_with_followup = input_with_followup + follow_up + ' \n Intermediate Answer: ' + intermediate_answer + ' \n '
                output_list = self.llm.generate(input_with_followup)
                Output = output_list[0]
                follow_up = Output.text
            # end of while
            if 'So the final answer is:' in follow_up: 
                follow_up = self._extract_final_answer_1(follow_up)
            elif 'Final Answer:' in follow_up:
                follow_up = self._extract_final_answer_2(follow_up)
            elif follow_up == '':
                # some special case will generate ''. In this situation we need add instruction for self ask finish the whole inference
                output_list = self.llm.generate(input_with_followup + 'So the final answer is:')
                Output = output_list[0]
                follow_up = Output.text
            else:
                print(f'Wrong final answer pattern!!!')
        else:
            passages = self.retrieval.search(query)
            passages = self._truncate_passages(passages)
            collated_passages = self.collate_passages(passages)
            target_instruction = self.find_instruction('self_ask-read', self.task)
            input = target_instruction.format_map({'passages': collated_passages, 'query': query})
            output_list = self.llm.generate(input)
            Output = output_list[0]
            follow_up = Output.text

            generation_track['cite passages'] = passages
        generation_track['final answer'] = follow_up # 
        final_answer = follow_up
        return final_answer, generation_track

    def _extract_followup(self, followup):
        followup_pattern = r'Follow up: (.+)'
        result = re.findall(followup_pattern, followup)
        followup_question = ''
        if len(result) >= 1:
            followup_question = result[0]
        return followup_question

    def _extract_final_answer_1(self, followup):
        followup_pattern = r'So the final answer is: (.+)'
        result = re.findall(followup_pattern, followup, re.DOTALL) # re.DOTALL flag instructs the regex engine to allow . to match any character, including newline \n.
        followup_question = ''
        if len(result) >= 1:
            followup_question = result[0]
        return followup_question
    
    def _extract_final_answer_2(self, followup):
        followup_pattern = r'Final Answer: (.+)'
        result = re.findall(followup_pattern, followup, re.DOTALL) # re.DOTALL flag instructs the regex engine to allow . to match any character, including newline \n.
        followup_question = ''
        if len(result) >= 1:
            followup_question = result[0]
        return followup_question