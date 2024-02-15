from raglab.dataset import  PopQA

class TriviaQA(PopQA):
    def __init__(self, output_dir, llm_path, eval_datapath):
        super().__init__(output_dir, llm_path, eval_datapath)
