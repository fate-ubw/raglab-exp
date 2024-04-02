from raglab.dataset.PopQA import  PopQA

class InputStruction:
    question:str
    answer:str

class OutputStruction:
    question:str
    answer:str
    generation:str

class StrategyQA(PopQA):
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
