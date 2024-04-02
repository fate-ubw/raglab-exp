from abc import ABC, abstractmethod

class MultiChoiceQA(ABC):
    def __init__(self, output_dir, llm_path, eval_datapath, eval_train_datapath=None):
        self.output_dir = output_dir
        self.llm_path = llm_path
        self.eval_datapath = eval_datapath
        self.eval_train_datapath = eval_train_datapath

    @abstractmethod
    def load_dataset(self): # The class that inherits the class must override load_dataset() methods
        pass
    
    @abstractmethod
    def save_result(self):# The class that inherits the class must override save_inference_result() methods
        pass
    
    @abstractmethod
    def get_instruction(self):
        pass
    
    def preprocess(self,input):
        return input