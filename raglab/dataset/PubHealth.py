import os
import jsonlines
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np

from raglab.dataset.utils import load_jsonlines
from raglab.dataset.metric import match
from raglab.dataset.base_dataset import MultiChoiceQA

class InputStruction:
    question:str
    answer:str
    pregiven_passages:str

class OutputStruction:
    question:str
    answer:str
    generation:str

TASK_INSTRUCTION = "Is the following statement correct or not? Say true if it's correct; otherwise say false."

PROMPT_INSTRUCTION = "### Instruction:\n{instruction}\n\n### Response:\n"

class PubHealth(MultiChoiceQA):
    def __init__(self, output_dir, llm_path, eval_datapath):
        super().__init__(output_dir, llm_path, eval_datapath)
        self.set_data_struction()

    def set_data_struction(self):
        '''
        The goal of constructing InputStruction and OutputStruction is to achieve the separation of algorithm logic and data, 
        so that users only need to add new dataset structures according to the rules without modifying the algorithm logic.
        '''
        self.inputStruction = InputStruction
        self.inputStruction.question = 'question'
        self.inputStruction.answer = 'answers'
        self.inputStruction.pregiven_passages = 'ctxs'
        self.outputStruction = OutputStruction
        self.outputStruction.question = 'question'
        self.outputStruction.answer = 'answers'
        self.outputStruction.generation = 'generation'

    def get_instruction(self, prompt):
        if len(TASK_INSTRUCTION) > 0:
            prompt = TASK_INSTRUCTION + "\n\n## Input:\n\n" + prompt
        prompt_with_instruction = PROMPT_INSTRUCTION.format_map({"instruction": prompt})
        return prompt_with_instruction

    def load_dataset(self): 
        if self.eval_datapath.endswith(".json"):
            eval_dataset = json.load(open(self.eval_datapath))
        else:
            eval_dataset = load_jsonlines(self.eval_datapath)
        # the type of eval_dataset is list of dict
        return eval_dataset

    def save_result(self, inference_result: list[dict]): 
        print('storing result....')
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        model_name = os.path.basename(self.llm_path)
        input_filename = os.path.basename(self.eval_datapath)
        eval_Dataname = os.path.splitext(input_filename)[0]
        time = datetime.now().strftime('%m%d_%H%M')
        output_name = f'infer_output-{eval_Dataname}-{model_name}-{time}.jsonl'
        output_file = os.path.join(self.output_dir, output_name)
        
        with jsonlines.open(output_file, 'w') as outfile: 
            outfile.write(inference_result)
        print(f'output file path:{output_file}')
        print('success!')

    def record_result(self, eval_data, final_prediction, inference_results):
        inference_results.append({
            'question': eval_data[self.inputStruction.question],
            'answers': eval_data['answers'],
            'generation': final_prediction})
        return inference_results

    def eval_acc(self, infer_results: list[dict]):
        print('start evaluation!')
        eval_results = []
        for idx, data in enumerate(infer_results):
            metric_result = match(data["generation"], data["answers"])
            eval_results.append(metric_result)
        # TODO save result in ***.json.eval_result file 
        return np.mean(eval_results)


