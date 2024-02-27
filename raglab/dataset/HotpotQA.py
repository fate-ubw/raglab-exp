import os
import jsonlines
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np

from raglab.dataset.utils import load_jsonlines
from raglab.dataset.metric import match
from raglab.dataset.base_dataset import QA

from dspy.datasets import HotPotQA

class HotpotQA(QA):
    def __init__(self, output_dir, llm_path, eval_datapath, rag):
        self.output_dir = output_dir
        self.llm_path = llm_path
        self.eval_datapath = eval_datapath
        self.rag = rag

    def load_dataset(self):
        try:
            if self.rag == "selfrag":
                print(f"\nCurrent RAG Method: {self.rag}\n")
                if self.eval_datapath.endswith(".json"):
                    eval_dataset = json.load(open(self.eval_datapath))
                else:
                    eval_dataset = load_jsonlines(self.eval_datapath)
                return eval_dataset
            elif self.rag == "dsp":
                print(f"\nCurrent RAG Method: {self.rag}\n")
                return HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
            else:
                raise ValueError(f"Invalid RAG : {self.model_mode}. Must be 'HFModel' or 'OpenAI'.")
        except ValueError as e:
            print(e)

    def save_result(self, inference_result: list[dict])-> None: 
        print('storing result....')
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        model_name = os.path.basename(self.llm_path)
        input_filename = os.path.basename(self.eval_datapath)
        eval_Dataname = os.path.splitext(input_filename)[0]
        time = datetime.now().strftime('%m%d_%H%M')
        output_name = f'infer_output-{eval_Dataname}-{model_name}-{time}.jsonl'
        output_file = os.path.join(self.output_dir, output_name)
        
        with jsonlines.open(output_file, 'w') as outfile: 
            outfile.write(inference_result)
        print(f'output file path:{output_file}')
        print('success!')

    def eval_acc(self, infer_results: list[dict]) -> float:
        print('start evaluation!')
        eval_results = []
        for idx, data in enumerate(tqdm(infer_results)):
            metric_result = match(data["generation"], data["answers"])
            eval_results.append(metric_result)
        # TODO 这里应该把结果存储下来***.json.eval_result
        return float(np.mean(eval_results))

    def get_instruction(self):
        pass
    
    def preprecess(self):
        pass


    def postprecess(self):
        pass
