import os
import argparse
from datetime import datetime
import pdb
from tqdm import tqdm
import json
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from raglab.dataset.utils import get_dataset # load popqa class
from raglab.rag.infer_alg.naive_rag.utils import load_evaldataset, save_inference_result
from raglab.retrieval.colbert.colbert_retrieve import ColbertRetrieve
from raglab.retrieval.contriever.contriever_retrieve import ContrieverRrtieve
import pudb
class NaiveRag:
    def __init__(self, args):
        self.args = args 
        self.task = args.task
        self.llm_path = args.llm_path # __init__ 只进行参数的传递，尤其是传递路径什么的
        self.generate_maxlength = args.generate_maxlength
        self.use_vllm = args.use_vllm
        self.eval_datapath = args.eval_datapath
        self.output_dir = args.output_dir
        
        # retrieval args
        self.retrieval_name = args.retrieval_name

        # setup model and database 
        self.llm, self.tokenizer, self.sampling_params = self.load_llm()
        self.retrieval = self.setup_retrieval()

    def init(self):
        
        raise NotImplementedError

    def inference(self, query: Optional[str] = None, mode = 'interact'):# mode 不会冲突因为这个mode 是函数内在的 mode
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            passages = self.retrieval.search(query)
            # passages: dict of dict
            inputs = self.get_instruction(passages, query) 
            outputs = self.llm_inference(inputs) 
            return outputs
        elif 'evaluation' == mode:
            pu.db
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset() # right
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                temp = {}
                question = eval_data["question"] # 这个参数是和具体数据相关的，这个 key 选什么也没有什么办法，到时候放到 dataset 里面
                passages = self.retrieval.search(question) # 这里面必须调用 search 函数因为 每个self.retrieval自带的search 函数都不一样没法统一
                inputs = self.get_instruction(passages, question)
                outputs = self.llm_inference(inputs)
                temp["question"] = question
                temp["answets"] = eval_data["answers"]
                temp["generation"] = outputs 
                inference_results.append(temp)
            
            self.EvalData.save_result(inference_results)
            eval_result = self.EvalData.eval_acc(inference_results) 
            print(f'PopQA accuracy: {eval_result}')
        return eval_result 
    
    def load_llm(self):
        llm = None
        tokenizer = None
        sampling_params = None
        if self.use_vllm:
            llm = LLM(model=self.llm_path) 
            sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_path, skip_special_tokens=False) #
            llm = AutoModelForCausalLM.from_pretrained(self.llm_path)
        return llm, tokenizer, sampling_params
    
    def setup_retrieval(self):
        if 'colbert' == self.retrieval_name:
            retrieval_model = ColbertRetrieve(self.args) 
            retrieval_model.setup_retrieve()
        elif 'contriever' == self.retrieval_name:
            retrieval_model = ContrieverRrtieve(self.args)
            retrieval_model.setup_retrieve()
        return retrieval_model 
    
    def get_instruction(self, passages, query): 
        # passages is dict type
        collater = ''
        for rank_id, tmp in passages.items(): 
            collater += f'Passages{rank_id}: ' + tmp['content'] +'\n'  # 这个拿回来之后             
        instruction = f'''
                [Task]
                Please answer the question based on the user's input context and comply with the answering requirements.
                [Background Knowledge]
                {collater}
                [Answering Requirements]
                - You need to strictly answer based on the content of the background knowledge, and it is forbidden to answer questions based on common sense and known information.
                - For information that is not known, simply answer "No relevant answer found"
                [Question]
                {query}
                '''
        # 感觉这块可以添加不同任务的 instruction 因为不同任务使用的instruction 是不一样的
        return instruction
    
    def llm_inference(self, inputs): 
        if self.use_vllm:
            output = self.llm.generate(inputs, self.sampling_params)
            output_text = output[0].outputs[0].text
        else:
            input_ids = self.tokenizer.encode(inputs, return_tensors="pt")
            output_ids = self.llm.generate(input_ids, do_sample = False, max_length = self.generate_maxlength)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens = False)
        if '<\s>' in output_text: 
            return output_text.replace("<s> " + inputs, "").replace("</s>", "").strip()
        else:
            return output_text.replace("<s> " + inputs, "").strip()