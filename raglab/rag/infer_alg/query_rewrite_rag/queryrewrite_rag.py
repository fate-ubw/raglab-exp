from typing import Optional, Any
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pdb
import pudb
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag

class queryRewrite_rag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, query: Optional[str] = None, mode = 'interact'):# mode 不会冲突因为这个mode 是函数内在的 mode
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            # rewrite the query for better retrieval 
            rewrite_query = self.rewrite(query)
            passages = self.retrieval.search(rewrite_query)
            # passages: dict of dict
            inputs = self.get_instruction(passages, query) 
            outputs = self.llm_inference(inputs) 
            return outputs
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset() # right
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                temp = {}
                question = eval_data["question"]
                # rewrite the query 
                rewrite_query = self.rewrite(query)
                passages = self.retrieval.search(rewrite_query)
                inputs = self.get_instruction(passages, question)
                outputs = self.llm_inference(inputs)
                temp["question"] = question
                temp["answets"] = eval_data["answers"]
                temp["generation"] = outputs 
                inference_results.append(temp)

            self.EvalData.save_result(inference_results)
            eval_result = self.EvalData.eval_acc(inference_results) 
            print(f'{self.task} Accuracy: {eval_result}')
        return eval_result 

    def rewrite(self, query):
        # 这里得套一个 instruction 然后得到结果
        query = self.get_instruction() # 但是这里面的 instruction 
        rewrite_query = self.llm.generate(query)
        return rewrite_query
    
    def get_instruction(self, passages, query):
        return 
