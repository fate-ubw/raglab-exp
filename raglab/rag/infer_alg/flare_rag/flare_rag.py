from typing import Optional, Any
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pdb
import pudb
import re
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag

class FlareRag(NaiveRag):
    def __init__(self,args):
        super().__init__(args)
        self.init(args)
    
    def init(self, args):
        self.beta = args.beta
        self.theta = args.theta

    def inference(self, query: Optional[str] = None, mode = 'interact'):# mode 不会冲突因为这个mode 是函数内在的 mode
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            if self.realtime_retrieval:
                passages = self.retrieval.search(query) #self.retrieval.search(query) -> dict[int,dict]
                collated_passages = self.collate_passages(passages)
                target_instruction = self.find_instruction('Naive_rag-interact', '')
                inputs = target_instruction.format_map({'passages': collated_passages, 'query': query})
            else:
                target_instruction = self.find_instruction('Naive_rag-interact-without_retrieval', '')
                inputs = target_instruction.format_map({'query': query})
            outputs = self.llm_inference(inputs)
            return outputs
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset()
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                question = eval_data[self.EvalData.inputStruction.question] 
                if self.realtime_retrieval:
                    passages = self.retrieval.search(question) #self.retrieval.search(query) -> dict[int,dict] 
                    # passages: dict of dict
                    collated_passages = self.collate_passages(passages)
                    target_instruction = self.find_instruction('Naive_rag-evaluation', self.task)
                    inputs = target_instruction.format_map({'passages': collated_passages, 'query': question})
                else:
                    target_instruction = self.find_instruction('Naive_rag-evaluation-without_retrieval', self.task)
                    inputs = target_instruction.format_map({'query': question})
                outputs = self.llm_inference(inputs)
                inference_results = self.EvalData.record_result(eval_data, outputs, inference_results)
                eval_result = self.EvalData.eval_acc(inference_results)
                print(f'{self.task} Accuracy in {idx} turn: {eval_result}')
            # end of for loop
            self.EvalData.save_result(inference_results)
            eval_result = self.EvalData.eval_acc(inference_results) 
            print(f'{self.task} Accuracy: {eval_result}')
        return eval_result 
