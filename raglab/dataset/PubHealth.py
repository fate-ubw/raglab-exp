import os
import jsonlines
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np

from raglab.dataset.utils import load_jsonlines
from raglab.dataset.metrics import match, exact_match, F1
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
    def __init__(self, output_dir, llm_path, eval_datapath, eval_train_datapath):
        super().__init__(output_dir, llm_path, eval_datapath, eval_train_datapath)
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

    def load_dataset(self)-> list[dict]:
        if self.eval_datapath.endswith(".json"):
            eval_dataset = json.load(open(self.eval_datapath))
        else:
            eval_dataset = load_jsonlines(self.eval_datapath)
        return eval_dataset

    def save_result(self, inference_result: list[dict])-> None: 
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

    def record_result(self, eval_data:dict, final_prediction:str, inference_results:list) -> list[dict]:
        inference_results.append(
            {
             self.outputStruction.question: eval_data[self.inputStruction.question],
             self.outputStruction.answer: eval_data[self.inputStruction.answer],
             self.outputStruction.generation: final_prediction
            })
        return inference_results
    
    def get_instruction(self, prompt:str) ->str:
        if len(TASK_INSTRUCTION) > 0:
            prompt = TASK_INSTRUCTION + "\n\n## Input:\n\n" + prompt
        prompt_with_instruction = PROMPT_INSTRUCTION.format_map({"instruction": prompt})
        return prompt_with_instruction


    def eval_acc(self, infer_results: list[dict]):
        print('start evaluation!')
        eval_results = []
        for idx, data in enumerate(infer_results):
            if type(data[self.outputStruction.answer]) is str:
                answer = [data[self.outputStruction.answer]]
            elif type(data[self.outputStruction.answer]) is list:
                answer = data[self.outputStruction.answer]
            else:
                raise InvalidAnswerType("The type of answer is invalid. Only str and list[str] is valid. Check the answer in your raw data.")
            metric_result = match(data[self.outputStruction.generation], answer)
            eval_results.append(metric_result)
        # TODO save result in ***.json.eval_result file 
        return np.mean(eval_results)
    def eval_exact_match(self, infer_results: list[dict]) -> float:
        print('Start calcualte exact match!')
        eval_reaults = []
        for _, data in enumerate(infer_results):
            if type(data[self.outputStruction.answer]) is str:
                answer = [data[self.outputStruction.answer]]
            elif type(data[self.outputStruction.answer]) is list:
                answer = data[self.outputStruction.answer]
            else:
                raise InvalidAnswerType("The type of answer is invalid. Only str and list[str] is valid. Check the answer in your raw data.")
            metric_result = exact_match(data[self.outputStruction.generation], answer)
            eval_reaults.append(metric_result)
        # TODO 这里应该把结果存储下来***.json.eval_result 
        return float(np.mean(eval_reaults))

    def eval_f1_score(self, infer_results: list[dict]) -> float:
        print('Start calcualte F1 score!')
        eval_reaults = []
        for _, data in enumerate(infer_results):
            if type(data[self.outputStruction.answer]) is str:
                answer = [data[self.outputStruction.answer]]
            elif type(data[self.outputStruction.answer]) is list:
                answer = data[self.outputStruction.answer]
            else:
                raise InvalidAnswerType("The type of answer is invalid. Only str and list[str] is valid. Check the answer in your raw data.")
            
            metric_result = F1(data[self.outputStruction.generation], answer)
            eval_reaults.append(metric_result)
        # TODO 这里应该把结果存储下来***.json.eval_result 
        return float(np.mean(eval_reaults))

class InvalidAnswerType(Exception):
    pass