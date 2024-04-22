import os
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Any
import logging
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.retrieval.colbert.colbert_retrieve import ColbertRetrieve
from raglab.retrieval.contriever.contriever_retrieve import ContrieverRrtieve
from raglab.language_model import OpenaiModel, HF_Model, HF_VLLM
from raglab.instruction_lab import INSTRUCTION_LAB
import pdb
class NaiveRag:
    def __init__(self, args):
        self.args = args
        # output file name config
        self.config = args.config
        self.algorithm_name = args.algorithm_name
        self.llm_path = args.llm_path
        self.llm_name = args.llm_name
        # eval config
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
        self.llm = self.steup_llm(args)
        # steup retrieval
        if self.realtime_retrieval:
            self.retrieval = self.setup_retrieval(args) # retrieval model
        if self.task == '':
            self.print_fn = print
        else:
            self.print_fn = self.setup_logger(args)
        self.init(args)
        self.print_fn(f'Rag Parameter:{args}')

    def init(self, args):
        pass

    def inference(self, query: Optional[str] = None, mode = 'interact'):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            self.print_fn(f"Interactive mode: query = {query}")
            final_answer, generation_track = self.infer(query)
            return final_answer, generation_track
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self) # here we input self because dataset classed need self.print_fn
            self.eval_dataset = self.EvalData.load_dataset()
            self.print_fn(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                eval_data = self.EvalData.preprocess(eval_data) # some dataset need preprocess such as: arc_challenge
                question = eval_data[self.EvalData.InputStruction.question] 
                # infer
                outputs, generation_track = self.infer(question)
                inference_results = self.EvalData.record_result(eval_data, outputs, inference_results)
                self.print_fn(f'{self.task} in {idx+1} turn:\n Question:{question} \n Rag Output:{outputs} \n Answers: {eval_data[self.EvalData.InputStruction.answer]}')
                # calculate metric
                acc = self.EvalData.eval_acc(inference_results)
                EM = self.EvalData.eval_exact_match(inference_results)
                f1_score = self.EvalData.eval_f1_score(inference_results)
                self.print_fn(f'{self.task} in {idx+1} turn: \n Accuracy: {acc} \n Exact match:{EM} \n F1 score: {f1_score}')
            # --> end of for loop
            self.EvalData.save_result(inference_results)
            eval_result = {'Accuracy':acc, 'Exact match': EM, 'F1 score':f1_score}
            self.EvalData.save_evaluation_results(eval_result)
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

    def steup_llm(self, args):
        if self.llm_mode == 'HF_Model':
            if self.use_vllm:
                llm = HF_VLLM(args)
                llm.load_model() # load_model() will load local model and tokenizer  
            else:
                llm = HF_Model(args)
                llm.load_model() # load_model() will load local model and tokenizer
        elif self.llm_mode == 'Openai_api':
            llm = OpenaiModel(args)
            llm.load_model() # load_model() will load api configs and tiktoken
        else:
            raise LanguageModelError("Language model must be huggingface or openai api.")
        return llm
    
    def setup_retrieval(self, args):
        if 'colbert' == self.retrieval_name:
            retrieval_model = ColbertRetrieve(args)
            retrieval_model.setup_retrieve()
        elif 'contriever' == self.retrieval_name:
            retrieval_model = ContrieverRrtieve(args)
            retrieval_model.setup_retrieve()
        return retrieval_model 
    
    def collate_passages(self, passages:dict[int, Optional[dict]])-> str:
        collate = ''
        for rank_id, doc in passages.items(): 
            if doc is None:
                continue
            if 'title' in doc:
                collate += f'#Passages{rank_id}: ' '##Title: '+ doc['title'] + ' ##Content: ' + doc['text'] +'\n' 
            else:
                collate += f'#Passages{rank_id}: ' + doc['text'] +'\n'
        return collate

    def setup_logger(self, args):
        # set logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG) # Set the log level to DEBUG
        # build file_name based on args
        self.output_dir = os.path.join(args.output_dir, args.task)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.time = datetime.now().strftime('%m%d_%H%M')
        if args.llm_mode == 'HF_Model':
            model_name = os.path.basename(self.llm_path.rstrip('/'))
            self.file_name = args.algorithm_name + '|' + args.task + '|' + model_name + '|' + args.retrieval_name + '|'
        else:
            self.file_name = args.algorithm_name + '|' + args.task + '|' + args.llm_name + '|' + args.retrieval_name + '|'
        # Create log file handler
        log_file = 'rag_output-' + self.file_name + f'time={self.time}.log'
        log_file = os.path.join(self.output_dir, log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Set log format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger.info

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