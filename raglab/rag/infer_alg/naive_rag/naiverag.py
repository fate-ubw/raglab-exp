import os
from datetime import datetime
from tqdm import tqdm
import json
from pprint import pprint
from typing import Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.retrieval.colbert.colbert_retrieve import ColbertRetrieve
from raglab.retrieval.contriever.contriever_retrieve import ContrieverRrtieve
from raglab.language_model.openai_lm import OpenaiModel
import pdb
class NaiveRag:
    def __init__(self, args):
        self.args = args 
        self.task = args.task
        self.eval_datapath = args.eval_datapath
        self.output_dir = args.output_dir

        # llm config
        self.llm_mode = args.llm_mode
        self.llm_path = args.llm_path
        self.dtype = args.dtype
        self.use_vllm = args.use_vllm
        self.generate_maxlength = args.generate_maxlength
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.generation_stop = args.generation_stop # TODO 转义符号的问题
        if  self.generation_stop == '\\n':
            self.generation_stop = '\n'
        
        # retrieval args
        self.n_docs = args.n_docs
        self.retrieval_name = args.retrieval_name
        self.realtime_retrieval = args.realtime_retrieval

        # setup model and database 
        self.llm, self.tokenizer, self.sampling_params = self.load_llm()
        if self.realtime_retrieval:
            self.retrieval = self.setup_retrieval() # retrieval model
        self.init(args)

    def init(self, args):
        pass

    def inference(self, query: Optional[str] = None, mode = 'interact'):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            final_answer, generation_track = self.infer(query)
            return final_answer, generation_track
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset()
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                eval_data = self.EvalData.preprocess(eval_data) # some dataset need preprocess such as: arc_challenge
                question = eval_data[self.EvalData.inputStruction.question] 
                # infer
                outputs, generation_track = self.infer(question)
                inference_results = self.EvalData.record_result(eval_data, outputs, inference_results)
                print(f'output:{outputs} \n eval_data: {eval_data[self.EvalData.inputStruction.answer]}')
                # calculate metric
                acc = self.EvalData.eval_acc(inference_results)
                EM = self.EvalData.eval_exact_match(inference_results)
                f1_score = self.EvalData.eval_f1_score(inference_results)
                print(f'{self.task} in {idx} turn: \n Accuracy: {acc} \n Exact match:{EM} \n F1 score: {f1_score}')
            # end of for loop
            self.EvalData.save_result(inference_results)
            # calculate metric
            acc = self.EvalData.eval_acc(inference_results)
            EM = self.EvalData.eval_exact_match(inference_results)
            f1_score = self.EvalData.eval_f1_score(inference_results)
            pprint(f'{self.task} in {idx} turn: \n Accuracy: {acc} \n Exact match:{EM} \n F1 score: {f1_score}')
            eval_result = {'Accuracy':acc, 'Exact match': EM, 'F1 score':f1_score}
            return eval_result
        else:
            raise ModeNotFoundError("Mode must be interact or evaluation. Please provide a valid mode.")

    def infer(self, query: str)->tuple[str,dict[str,Any]]:
        '''
        infer function of naive rag
        '''
        generation_track = {}
        if self.realtime_retrieval:
            passages = self.retrieval.search(query) #self.retrieval.search(query) -> dict[int,dict]
            collated_passages = self.collate_passages(passages)
            target_instruction = self.find_instruction('Naive_rag', self.task)
            input = target_instruction.format_map({'passages': collated_passages, 'query': query})
            generation_track['cited passages'] = passages
        else:
            target_instruction = self.find_instruction('Naive_rag-without_retrieval', self.task)
            input = target_instruction.format_map({'query': query})
        outputs = self.llm_inference(input)
        generation_track['final answer'] = outputs
        return outputs, generation_track

    def load_llm(self):
        llm = None
        tokenizer = None
        sampling_params = None
        if self.llm_mode == 'HF_Model':
            if self.use_vllm:
                llm = LLM(model=self.llm_path, tokenizer=self.llm_path, dtype=self.dtype)
                if self.generation_stop != '':
                    sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, stop=[self.generation_stop], repetition_penalty= 1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
                else:
                    sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, repetition_penalty= 1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
                tokenizer = llm.get_tokenizer()
            else:
                if self.dtype == 'half' or self.dtype == 'float16':
                    llm = AutoModelForCausalLM.from_pretrained(self.llm_path, device_map="auto", torch_dtype=torch.float16)
                else:
                    llm = AutoModelForCausalLM.from_pretrained(self.llm_path, device_map="auto")
                tokenizer = AutoTokenizer.from_pretrained(self.llm_path, skip_special_tokens=False)
        elif self.llm_mode == 'Openai_api':
            llm = OpenaiModel(self.args)
            llm.load_model()
            tokenizer = llm.tokenizer
        else:
            raise LanguageModelError("Language model must be huggingface or openai api.")
        return llm, tokenizer, sampling_params
    
    def llm_inference(self, input:str)->str: 
        if self.llm_mode == 'HF_Model':
            if self.use_vllm:
                output = self.llm.generate(input, self.sampling_params)
                output_text = output[0].outputs[0].text
            else:
                input_ids = self.tokenizer.encode(input, return_tensors="pt")
                instruction_len = input_ids.shape[1]
                output_ids = self.llm.generate(input_ids, do_sample = False, max_length =instruction_len + self.generate_maxlength)
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens = False)
        elif self.llm_mode == 'Openai_api':
            # naive rag only need text
            Output = self.llm.generate(input)
            output_text = Output.text
        else:
            raise LanguageModelError("Language model must be huggingface or openai api.")
        # stop function cat the answer based on stop str
        if self.generation_stop != '':
            stop_index = output_text.find(self.generation_stop) # 为什么这里的\n 变成了\n
            if stop_index != -1:
                output_text = output_text[:stop_index].strip()
        
        # remove special tokens
        if '</s>' in output_text:
            return output_text.replace("<s> " + input, "").replace("</s>", "").strip()
        else:
            return output_text.replace("<s> " + input, "").strip()    
    
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
        for rank_id, doc in passages.items(): 
            if doc is None:
                continue
            if 'title' in doc:
                collate += f'#Passages{rank_id}: ' '##Title: '+ doc['title'] + ' ##Content: ' + doc['content'] +'\n' 
            else:
                collate += f'#Passages{rank_id}: ' + doc['content'] +'\n'
        return collate

    def find_instruction(self, rag_name:str, dataset_name:str) -> str:
        file_path = './instruction_lab/instruction_lab.json'
        target_instruction = ''
        with open(file_path, 'r') as file:
            instructions = json.load(file)
        for instruction in instructions:
            if instruction['rag_name'] == rag_name and instruction['dataset_name'] == dataset_name:
                target_instruction = instruction['instruction']
                break
        if target_instruction == '':
            raise InstructionNotFoundError('Instruction name not recognized. Please provide a valid instruction name.')
        return target_instruction

# custom Exceptions
class ModeNotFoundError(Exception):
    pass

class InstructionNotFoundError(Exception):
    pass

class LanguageModelError(Exception):
    pass