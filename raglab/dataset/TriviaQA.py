from raglab.dataset.PopQA import  PopQA

class TriviaQA(PopQA):
    def __init__(self, output_dir, llm_path, eval_datapath, eval_train_datapath):
        super().__init__(output_dir, llm_path, eval_datapath, eval_train_datapath)