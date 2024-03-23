import os
import argparse
from datetime import datetime
import pdb
from tqdm import tqdm
import json
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from raglab.dataset.utils import get_dataset # load dataset class
from raglab.rag.infer_alg.naive_rag.utils import load_evaldataset, save_inference_result
from raglab.retrieval.colbert.colbert_retrieve import ColbertRetrieve
from raglab.retrieval.contriever.contriever_retrieve import ContrieverRrtieve
import pudb
import pdb
class NaiveRag:
    def __init__(self, args):
        self.args = args 
        self.task = args.task
        self.eval_datapath = args.eval_datapath
        self.output_dir = args.output_dir

        # llm config
        self.llm_path = args.llm_path # __init__ 只进行参数的传递，尤其是传递路径什么的
        self.dtype = args.dtype
        self.use_vllm = args.use_vllm
        self.generate_maxlength = args.generate_maxlength
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.generation_stop = args.generation_stop

        # retrieval args
        self.n_docs = args.n_docs
        self.retrieval_name = args.retrieval_name
        self.realtime_retrieval = args.realtime_retrieval

        # setup model and database 
        self.llm, self.tokenizer, self.sampling_params = self.load_llm()
        if self.realtime_retrieval:
            self.retrieval = self.setup_retrieval() # retrieval model

    def init(self):

        raise NotImplementedError

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
    
    def load_llm(self):
        llm = None
        tokenizer = None
        sampling_params = None
        if self.use_vllm:
            llm = LLM(model=self.llm_path, tokenizer=self.llm_path, dtype=self.dtype)
            if self.generation_stop != '':
                sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, stop=[self.generation_stop], repetition_penalty= 1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
            else:
                sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, repetition_penalty= 1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
            tokenizer = llm.get_tokenizer()
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_path, skip_special_tokens=False)
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
    
    def collate_passages(self, passages:dict[int, Optional[dict]])-> str:
        collate = ''
        for rank_id, tmp in passages.items(): 
            if tmp is None:
                continue
            collate += f'Passages{rank_id}: ' + tmp['content'] +'\n'  
        return collate

    def llm_inference(self, inputs): 
        if self.use_vllm:
            output = self.llm.generate(inputs, self.sampling_params)
            output_text = output[0].outputs[0].text
            pdb.set_trace()
        else:
            pdb.set_trace()
            input_ids = self.tokenizer.encode(inputs, return_tensors="pt")
            instruction_len = input_ids.shape[1]
            output_ids = self.llm.generate(input_ids, do_sample = False, max_length =instruction_len + self.generate_maxlength)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens = False)
        if '</s>' in output_text:
            return output_text.replace("<s> " + inputs, "").replace("</s>", "").strip()
        else:
            return output_text.replace("<s> " + inputs, "").strip()

    def find_instruction(self, rag_name:str, dataset_name:str) -> str:
        file_path = './instruction_lab/instruction_lab.json'
        with open(file_path, 'r') as file:
            instructions = json.load(file)
        for instruction in instructions:
            if instruction['rag_name'] == rag_name and instruction['dataset_name'] == dataset_name:
                target_instruction = instruction['instruction']
                break
        return target_instruction

