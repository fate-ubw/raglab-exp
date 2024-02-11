import os
import argparse
from datetime import datetime
import pdb
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from raglab.dataset.PopQA import PopQA # load popqa class
from raglab.rag.infer_alg.utils import load_evaldataset, save_inference_result
from raglab.retrieval.colbert.colbert_retrieve import ColbertRetrieve
from raglab.retrieval.contriever.contriever_retrieve import ContrieverRrtieve
import pudb
class NaiveRag:
    def __init__(self, args):
        self.args = args 
        # common args
        self.llm_path = args.llm_path # __init__ 只进行参数的传递，尤其是传递路径什么的
        self.generate_maxlength = args.generate_maxlength
        self.use_vllm = args.use_vllm
        self.num_gpu = args.num_gpu

        self.eval_datapath = args.eval_datapath
        self.output_dir = args.output_dir
        
        # retrieval args
        # 这些 args 其实都可以删除的，因为在 Naive 类中没有存储下来
        self.retrieval_name = args.retrieval_name
        self.n_docs = args.n_docs
        self.nbits = args.nbits
        self.doc_maxlen = args.doc_maxlen# 后期根据数据的处理情况都定义下来
        self.retriever_modelPath = args.retriever_modelPath 
        self.index_dbPath = args.index_dbPath
        self.text_dbPath = args.text_dbPath

        # setup model and database 
        self.llm, self.tokenizer = self.load_llm()
        self.retrieval = self.setup_retrieval()

    def init(self):
        
        raise NotImplementedError

    def inference(self, query = None, mode = 'interact', task = None):
        assert mode in ['interact', 'evaluation']
        assert task in ['PopQA']
        if 'interact' == mode:
            passages = self.search(query)
            # passages: dict of dict
            inputs = self.get_prompt(passages, query) 
            outputs = self.llm_inference(inputs) # 
            response = self.postporcess(outputs) 
            return response
        elif 'evaluation' == mode:
            if 'PopQA' == task:
                popqa =  PopQA(self.output_dir, self.llm_path, self.eval_datapath) 
                self.eval_dataset = popqa.load_dataset() #
                
                inference_results = []
                for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                    question = eval_data["question"] # 这个参数是和具体数据相关的，这个 key 选什么也没有什么办法，到时候放到 dataset 里面
                    passages = self.search(question)
                    inputs = self.get_prompt(passages, question)
                    outputs = self.llm_inference(inputs)
                    eval_data["generation"] = outputs 
                    inference_results.append(eval_data)
                
                popqa.save_result(inference_results, self.output_dir)
                eval_result = popqa.eval_acc(inference_results) 
                print(f'PopQA accuracy: {eval_result}')
            return eval_result 

    def load_llm(self):
        llm = None
        tokenizer = None
        if self.use_vllm:
            llm = LLM(model=self.llm_path) 
            self.sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_path, skip_special_tokens=False) #
            llm = AutoModelForCausalLM.from_pretrained(self.llm_path)
        return llm, tokenizer
    
    def setup_retrieval(self):
        if 'colbert' == self.retrieval_name:
            retrieval_model = ColbertRetrieve(self.args) 
            retrieval_model.setup_retrieve()
        elif 'contriever' == self.retrieval_name:
            retrieval_model = ContrieverRrtieve(self.args)
            retrieval_model.setup_retrieve()
        return retrieval_model 
    
    def get_prompt(self, passages, query): 
        # passages is dict type
        collater = ''
        for rank_id, tmp in passages.items(): 
            collater += f'Passages{rank_id}: ' + tmp['content'] +'\n'  # 这个拿回来之后             
        prompt = f'''
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
        return prompt

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
    
    def search(self, query):
        if 'colbert' == self.retrieval_name:
            passages = self.retrieval.search(query)
        elif 'contriever' == self.retrieval_name:
            passages = self.retrieval.search(query)

        return passages