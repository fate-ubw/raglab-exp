from raglab.dataset.PubHealth import PubHealth

class ArcChallenge(PubHealth):
    def __init__(self, output_dir, llm_path, eval_datapath):
        super().__init__(output_dir, llm_path, eval_datapath)
        