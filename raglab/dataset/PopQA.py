import os
import jsonlines
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from raglab.dataset.utils import load_jsonlines
from raglab.dataset.metrics import match, exact_match, F1
from raglab.dataset.base_dataset import QA

TASK_INSTRUCTION = '' # open QA no need special instruction for inference

PROMPT_INSTRUCTION = "### Instruction:\n{instruction}\n\n### Response:\n"

class PopQA(QA):
    def __init__(self, output_dir, llm_path, eval_datapath, eval_train_datapath):
        super().__init__(output_dir, llm_path, eval_datapath, eval_train_datapath)
    
    @dataclass
    class InputStruction:
        question:str = 'question'
        answer:str = 'answers'
        pregiven_passages:str = 'ctxs' 

    @dataclass
    class OutputStruction:
        question:str = 'question'
        answer:str = 'answers'
        generation:str = 'generation'

    def load_dataset(self)-> list[dict]:
        if self.eval_datapath.endswith(".json"):
            eval_dataset = json.load(open(self.eval_datapath))
        else:
            eval_dataset = load_jsonlines(self.eval_datapath)
        return eval_dataset

    def save_result(self, inference_result: list[dict])-> None: 
        print('storing inference result....')
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
        inference_results.append(
            {
             self.OutputStruction.question: eval_data[self.InputStruction.question],
             self.OutputStruction.answer: eval_data[self.InputStruction.answer],
             self.OutputStruction.generation: final_prediction
            })
        return inference_results

    def get_instruction(self, prompt:str)->str:
        if len(TASK_INSTRUCTION) > 0:
            prompt = TASK_INSTRUCTION + "\n\n## Input:\n\n" + prompt
        prompt_with_instruction = PROMPT_INSTRUCTION.format_map({"instruction": prompt})
        return prompt_with_instruction

    def eval_acc(self, infer_results: list[dict]) -> float:
        print('start calculate accuracy!')
        eval_results = []
        for _, data in enumerate(infer_results):
            if type(data[self.OutputStruction.answer]) is str:
                answer = [data[self.OutputStruction.answer]]
            elif type(data[self.OutputStruction.answer]) is list:
                answer = data[self.OutputStruction.answer]
            elif type(data[self.OutputStruction.answer]) is bool: # The answer of StrategyQA is bool
                answer = [str(data[self.OutputStruction.answer])]
            else:
                raise InvalidAnswerType("The type of answer is invalid. Only str and list[str] is valid. Check the answer in your raw data.")
            metric_result = match(data[self.OutputStruction.generation], answer)
            eval_results.append(metric_result)
        # TODO 这里应该把结果存储下来***.json.eval_result 
        return float(np.mean(eval_results))

    def eval_exact_match(self, infer_results: list[dict]) -> float:
        print('Start calcualte exact match!')
        eval_reaults = []
        for _, data in enumerate(infer_results):
            if type(data[self.OutputStruction.answer]) is str:
                answer = [data[self.OutputStruction.answer]]
            elif type(data[self.OutputStruction.answer]) is list:
                answer = data[self.OutputStruction.answer]
            else:
                raise InvalidAnswerType("The type of answer is invalid. Only str and list[str] is valid. Check the answer in your raw data.")
            metric_result = exact_match(data[self.OutputStruction.generation], answer)
            eval_reaults.append(metric_result)
        # TODO 这里应该把结果存储下来***.json.eval_result 
        return float(np.mean(eval_reaults))

    def eval_f1_score(self, infer_results: list[dict]) -> float:
        print('Start calcualte F1 score!')
        eval_reaults = []
        for _, data in enumerate(infer_results):
            if type(data[self.OutputStruction.answer]) is str:
                answer = [data[self.OutputStruction.answer]]
            elif type(data[self.OutputStruction.answer]) is list:
                answer = data[self.OutputStruction.answer]
            else:
                raise InvalidAnswerType("The type of answer is invalid. Only str and list[str] is valid. Check the answer in your raw data.")
            
            metric_result = F1(data[self.OutputStruction.generation], answer)
            eval_reaults.append(metric_result)
        # TODO 这里应该把结果存储下来***.json.eval_result 
        return float(np.mean(eval_reaults))

class InvalidAnswerType(Exception):
    pass