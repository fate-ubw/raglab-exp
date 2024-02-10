import os
import jsonlines
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np

from raglab.dataset.utils import load_jsonlines
from raglab.dataset.metric import match
class PopQA:
    def __init__(self, output_dir, llm_path, eval_datapath):
        # init all the path of 
        self.output_dir = output_dir
        self.llm_path = llm_path # 这个和 dataset 本身没有什么关系，所以应该作为具体方法的参数传入进去
        self.eval_datapath = eval_datapath

    def load_dataset(self): # 
        if self.eval_datapath.endswith(".json"):
            eval_dataset = json.load(open(self.eval_datapath))
        else:
            eval_dataset = load_jsonlines(self.eval_datapath) # 这一部分拿到的是一个 list of dict 
    # eval_dataset：type：list of dict
        return eval_dataset
    
    def save_result(self, inference_result: list[dict], output_dir): # 这个还是直接使用吧
        print('storing result....')
        if not os.path.exists(output_dir): #这个参数也走 yaml 文件里面的吧
            os.makedirs(output_dir)
        model_name = os.path.basename(self.llm_path) #llm_path 这个能隐藏的就隐藏起来，因为在yaml 文件里面肯定是会使用
        input_filename = os.path.basename(self.eval_datapath) # 反正都要定义 PopQA 不如把这些参数都在 init 里面传了就完完了
        eval_Dataname = os.path.splitext(input_filename)[0] #这个拿到的是dataset 的 name
        time = datetime.now().strftime('%m%d_%H%M') # time 
        output_name = f'infer_output-{eval_Dataname}-{model_name}-{time}.jsonl' #
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
        for idx, data in enumerate(tqdm(infer_results)): #把全部的数据传进来，然后直接进行 for loop
            metric_result = match(data["generation"], data["answers"])
            eval_results.append(metric_result)
        # 这里应该把结果存储下来***.json.eval_result
        return np.mean(eval_results) 