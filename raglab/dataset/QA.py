from abc import ABC, abstractmethod

class QA(ABC):
    def __init__(self, output_dir, llm_path, eval_datapath):
        self.output_dir = output_dir
        self.llm_path = llm_path
        self.eval_datapath = eval_datapath
    
    @abstractmethod
    def load_dataset(self): # The class that inherits the class must override load_dataset() methods
        pass
    
    @abstractmethod
    def save_result(self):# The class that inherits the class must override save_inference_result() methods
        pass