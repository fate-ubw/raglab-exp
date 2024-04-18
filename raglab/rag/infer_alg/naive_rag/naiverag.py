import os
from tqdm import tqdm
from pprint import pprint
from typing import Optional, Any
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.retrieval.colbert.colbert_retrieve import ColbertRetrieve
from raglab.retrieval.contriever.contriever_retrieve import ContrieverRrtieve
from raglab.language_model import OpenaiModel, HF_Model, HF_VLLM
from raglab.instruction_lab import INSTRUCTION_LAB
import pdb
class NaiveRag:
    def __init__(self, args):
        self.args = args 
        self.task = args.task
        self.eval_datapath = args.eval_datapath
        self.output_dir = args.output_dir
        # llm config
            #(other configs check raglab/language_model/*.py files)
        self.llm_mode = args.llm_mode
        self.use_vllm = args.use_vllm
        # retrieval args
        self.n_docs = args.n_docs
        self.retrieval_name = args.retrieval_name
        self.realtime_retrieval = args.realtime_retrieval
        # setup model and database 
        self.llm = self.steup_llm()
        # steup retrieval
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
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset()
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                eval_data = self.EvalData.preprocess(eval_data) # some dataset need preprocess such as: arc_challenge
                question = eval_data[self.EvalData.InputStruction.question] 
                # infer
                outputs, generation_track = self.infer(question)
                inference_results = self.EvalData.record_result(eval_data, outputs, inference_results)
                print(f'output:{outputs} \n eval_data: {eval_data[self.EvalData.IutputStruction.answer]}')
                # calculate metric
                acc = self.EvalData.eval_acc(inference_results)
                EM = self.EvalData.eval_exact_match(inference_results)
                f1_score = self.EvalData.eval_f1_score(inference_results)
                print(f'{self.task} in {idx} turn: \n Accuracy: {acc} \n Exact match:{EM} \n F1 score: {f1_score}')
            # --> end of for loop
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
        outputs_list = self.llm.generate(input) # llm.generate() -> list[BaseOutputs] so you have to get the text from BaseOutputs.text
        Outputs = outputs_list[0]
        outputs_text = Outputs.text
        generation_track['final answer'] = outputs_text
        return outputs_text, generation_track

    def steup_llm(self):
        if self.llm_mode == 'HF_Model':
            if self.use_vllm:
                llm = HF_VLLM(self.args)
                llm.load_model() # load_model() will load local model and tokenizer  
            else:
                llm = HF_Model(self.args)
                llm.load_model() # load_model() will load local model and tokenizer
        elif self.llm_mode == 'Openai_api':
            llm = OpenaiModel(self.args)
            llm.load_model() # load_model() will load api configs and tiktoken
        else:
            raise LanguageModelError("Language model must be huggingface or openai api.")
        return llm
    
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
        for instruction in INSTRUCTION_LAB:
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