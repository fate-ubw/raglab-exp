from raglab.dataset.PopQA import  PopQA
from dataclasses import dataclass

class StrategyQA(PopQA):
    def __init__(self, output_dir, llm_path, eval_datapath, eval_train_datapath):
        super().__init__(output_dir, llm_path, eval_datapath, eval_train_datapath)

    @dataclass
    class InputStruction:
        question:str =  'question'
        answer:str = 'answer'

    @dataclass
    class OutputStruction:
        question:str = 'question'
        answer:str = 'answer' 
        generation:str = 'generation'