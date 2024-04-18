from raglab.dataset.PubHealth import PubHealth
from dataclasses import dataclass

class Feverous(PubHealth):
    def __init__(self, output_dir, llm_path, eval_datapath, eval_train_datapath):
        super().__init__(output_dir, llm_path, eval_datapath, eval_train_datapath)

    @dataclass
    class InputStruction:
        '''
        The goal of constructing InputStruction and OutputStruction is to achieve the separation of algorithm logic and data, 
        so that users only need to rewrite  set_data_struction() without modifying the algorithm logic.
        '''
        question:str = 'claim'
        answer:str = 'label'

    @dataclass
    class OutputStruction:
        question:str = 'claim'
        answer:str = 'label'
        generation:str = 'generation'