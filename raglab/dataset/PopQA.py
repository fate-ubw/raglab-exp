import os
import jsonlines
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np

from raglab.dataset.utils import load_jsonlines
from raglab.dataset.metric import match
from raglab.dataset.QA import QA
class PopQA(QA):
    def __init__(self, output_dir, llm_path, eval_datapath):
        super().__init__(output_dir, llm_path, eval_datapath)
    
    def load_dataset(self): 
        if self.eval_datapath.endswith(".json"):
            eval_dataset = json.load(open(self.eval_datapath))
        else:
            eval_dataset = load_jsonlines(self.eval_datapath)
    # eval_dataset：type：list of dict
        return eval_dataset
    
    def save_result(self, inference_result: list[dict], output_dir): # 这个还是直接使用吧
        print('storing result....')
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
        model_name = os.path.basename(self.llm_path)
        input_filename = os.path.basename(self.eval_datapath)
        eval_Dataname = os.path.splitext(input_filename)[0]
        time = datetime.now().strftime('%m%d_%H%M')
        output_name = f'infer_output-{eval_Dataname}-{model_name}-{time}.jsonl'
        output_file = os.path.join(output_dir, output_name)
        
        with open(output_file, 'w') as outfile:
            for result in inference_result:
                json.dump(result, outfile)
                outfile.write('\n')
        print(f'output file path:{output_file}')
        print('success!')

    def eval_acc(self, infer_results: list[dict]): #
        print('start evaluation!')
        eval_results = []
        for idx, data in enumerate(tqdm(infer_results)):
            metric_result = match(data["generation"], data["answers"])
            eval_results.append(metric_result)
        # TODO 这里应该把结果存储下来***.json.eval_result
        return np.mean(eval_results)