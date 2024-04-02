import pdb
from raglab.dataset.PopQA import PopQA
import numpy as np
from raglab.dataset.metrics import HotPotF1

class InputStruction:
    question:str
    answer:str

class OutputStruction:
    question:str
    answer:str
    generation:str

class HotPotQA(PopQA):
    def __init__(self, output_dir, llm_path, eval_datapath, eval_train_datapath):
        super().__init__(output_dir, llm_path, eval_datapath, eval_train_datapath)

    def set_data_struction(self):
        '''
        The goal of constructing InputStruction and OutputStruction is to achieve the separation of algorithm logic and data, 
        so that users only need to rewrite  set_data_struction() without modifying the algorithm logic.
        '''
        self.inputStruction = InputStruction
        self.inputStruction.question = 'question'
        self.inputStruction.answer = 'answer'

        self.outputStruction = OutputStruction
        self.outputStruction.question = 'question'
        self.outputStruction.answer = 'answer'
        self.outputStruction.generation = 'generation'
    
    def eval_f1_score(self, infer_results: list[dict]) -> float:
        '''
        the HotpotQA need to preprocess specific cases for 'yes', 'no', and 'noanswer' predictions.
        '''
        print('Start calcualte F1 score!')
        eval_reaults = []
        for _, data in enumerate(infer_results):
            if type(data[self.outputStruction.answer]) is str:
                answer = [data[self.outputStruction.answer]]
            elif type(data[self.outputStruction.answer]) is list:
                answer = data[self.outputStruction.answer]
            else:
                raise InvalidAnswerType("The type of answer is invalid. Only str and list[str] is valid. Check the answer in your raw data.")
            
            metric_result = HotPotF1(data[self.outputStruction.generation], answer)
            eval_reaults.append(metric_result)
        # TODO 这里应该把结果存储下来***.json.eval_result 
        return float(np.mean(eval_reaults))

class InvalidAnswerType(Exception):
    pass