from typing import Optional, Any
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pdb
import pudb
import re
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag

class QueryRewrite_rag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, query: Optional[str] = None, mode = 'interact'):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            # rewrite the query
            instruction = self.find_instruction('query_rewrite_rag-rewrite-Multiple_choice_QA', self.task) # name 再+一个Multiple_choice_QA即可
            query_with_instruction = instruction.format_map({'query':query})
            rewrite_query = self.rewrite(query_with_instruction)
            pattern = r"Query :(.*?)\*\*"
            matches = re.findall(pattern, rewrite_query)
            if len(matches) > 0:
                rewrite_query = matches[0]
            # retrieval
            passages = self.retrieval.search(rewrite_query)
            collated_passages = self.collate_passages(passages)
            # read
            instruction = self.find_instruction('query_rewrite_rag-read', self.task)
            query_with_instruction = instruction.format_map({'query':query, 'passages':collated_passages})
            outputs = self.llm_inference(query_with_instruction)
            return outputs
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset() # right
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                question = eval_data[self.EvalData.inputStruction.question]
                # rewrite the query
                instruction = self.find_instruction('query_rewrite_rag-rewrite-Multiple_choice_QA', self.task)
                query_with_instruction = instruction.format_map({'query':question})
                rewrite_query = self.rewrite(query_with_instruction)
                pattern = r"Query :(.*?)\*\*"
                matches = re.findall(pattern, rewrite_query)
                if len(matches) > 0:
                    rewrite_query = matches[0]
                # retrieval
                passages = self.retrieval.search(rewrite_query)
                collated_passages = self.collate_passages(passages)
                # read
                instruction = self.find_instruction('query_rewrite_rag-read', self.task)
                query_with_instruction = instruction.format_map({'query':question, 'passages':collated_passages})
                outputs = self.llm_inference(query_with_instruction)
                inference_results = self.EvalData.record_result(eval_data, outputs, inference_results)

            self.EvalData.save_result(inference_results)
            eval_result = self.EvalData.eval_acc(inference_results) 
            print(f'{self.task} Accuracy: {eval_result}')
        return eval_result 
    
    def rewrite(self, query):
        rewrite_query = self.llm_inference(query)
        return rewrite_query

