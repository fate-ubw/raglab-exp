import os
import jsonlines
from raglab.dataset.PopQA import  PopQA
from datetime import datetime
from dataclasses import dataclass


TASK_INSTRUCTION = "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."

PROMPT_INSTRUCTION = "### Instruction:\n{instruction}\n\n### Response:\n"

class ASQA(PopQA):
    def __init__(self, output_dir, llm_path, eval_datapath, eval_train_datapath):
        super().__init__(output_dir, llm_path, eval_datapath, eval_train_datapath)

    @dataclass
    class InputStruction:
        '''
        The goal of constructing InputStruction and OutputStruction is to achieve the separation of algorithm logic and data, 
        so that users only need to add new dataset structures according to the rules without modifying the algorithm logic.
        '''
        question:str = 'question'
        answer:str = 'answer'
        pregiven_passages:str = 'docs'

    @dataclass
    class OutputStruction:
        question:str = 'question'
        answer:str = 'answer'
        generation:str = 'output'
        cite_passages:str = 'docs'
        generation_track:str = 'intermediate'

    def save_result(self, inference_result: list[dict])-> None: 
        print('storing result....')
        new_results = {"data": inference_result, "args": [], "total_cost": 0.0, "azure_filter_fail": ""}
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        model_name = os.path.basename(self.llm_path)
        input_filename = os.path.basename(self.eval_datapath)
        eval_Dataname = os.path.splitext(input_filename)[0]
        time = datetime.now().strftime('%m%d_%H%M')
        output_name = f'infer_output-{eval_Dataname}-{model_name}-{time}.jsonl'
        output_file = os.path.join(self.output_dir, output_name)
        with jsonlines.open(output_file, 'w') as outfile: 
            outfile.write(new_results)
        print(f'output file path:{output_file}')
        print('success!')

    def get_instruction(self, prompt):
        if len(TASK_INSTRUCTION) > 0:
            prompt = TASK_INSTRUCTION + "## Input:\n\n" + prompt
        prompt_with_instruction = PROMPT_INSTRUCTION.format_map({"instruction": prompt})
        return prompt_with_instruction

    def record_result(self, eval_data, final_prediction_with_citation, catation_docs, response_id, generation_track, inference_results):
        '''
        - record inference results 
        '''
        if "original_splitted_sentences" in generation_track:
            inference_results.append(
                {
                    self.OutputStruction.question: eval_data[self.InputStruction.question],
                    self.OutputStruction.answer: eval_data[self.InputStruction.answer],
                    self.OutputStruction.generation: final_prediction_with_citation[response_id],
                    self.OutputStruction.cite_passages: catation_docs[response_id],
                    self.OutputStruction.generation_track: generation_track['original_splitted_sentences'][response_id]
                }
                    )
        else:
            inference_results.append(
                {
                    self.OutputStruction.question: eval_data[self.InputStruction.question],
                    self.OutputStruction.answer: eval_data[self.InputStruction.answer],
                    self.OutputStruction.generation: final_prediction_with_citation[response_id],
                    self.OutputStruction.cite_passages: catation_docs[response_id]
                }
                    )
        return inference_results
