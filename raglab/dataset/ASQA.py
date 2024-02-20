from raglab.dataset.PopQA import  PopQA

class ASQA(PopQA):
    def __init__(self, output_dir, llm_path, eval_datapath):
        super().__init__(output_dir, llm_path, eval_datapath)
