import sys
import os
import pdb
from tqdm import tqdm
import dspy
from dspy import Example
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
import dspy.evaluate.metrics as Metrics
from dspy.datasets import HotPotQA
from raglab.dataset.utils import get_dataset
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
import raglab.rag.infer_alg.dsp.utils as dsp_utils

class Dsp(NaiveRag):
    def __init__(self, args):
        self.args = args 
        self.task = args.task
        self.eval_datapath = args.eval_datapath
        self.eval_train_datapath = args.eval_train_datapath
        self.output_dir = args.output_dir

        # llm config
        self.llm_path = args.llm_path
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

        # api or local LLM
        self.model_mode = args.model_mode
        # openai parameters
        self.llm_api = args.llm_api
        self.api_key = args.api_key
        self.api_base = args.api_base

        # setup model and database 
        self.llm = self.load_llm()
        if self.realtime_retrieval:
            self.retrieval = self.setup_retrieval() # retrieval model
        self.init(args)

    def init(self, args):
        # dsp parameters
        self.inference_CoT = args.inference_CoT
        self.signature_retrieval = args.signature_retrieval
        self.max_hops = args.max_hops
        self.eval_threads = args.eval_threads
        # evaluate parameters
        self.mrtrics = args.metrics
        dspy.settings.configure(lm=self.llm, rm=self.retrieval) 

    def inference(self, query='', mode='interact'):
        assert mode in ['interact', 'evaluation']
        self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath, self.eval_train_datapath)
        train_dataset = self.EvalData.load_train_dataset()
        train_dataset = self.dic2Example(train_dataset) # transfer list[dict] -> list[Example]
        # define model
        teleprompter = BootstrapFewShot(metric=dsp_utils.validate_context_and_answer_and_hops) 
        compiled_rag = teleprompter.compile(dsp_utils.SimplifiedBaleen(retrieve=self.retrieval), teacher=dsp_utils.SimplifiedBaleen(retrieve=self.retrieval), trainset=train_dataset) 
        
        if 'interact' == mode:
            outputs = compiled_rag(query)
            return outputs['answer'] 
        elif 'evaluation' == mode:
            self.eval_dataset = self.EvalData.load_dataset()
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            pdb.set_trace()
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                question = eval_data[self.EvalData.inputStruction.question]
                final_generation = compiled_rag(question)
                final_generation = final_generation[self.EvalData.outputStruction.answer]
                inference_results = self.EvalData.record_result(eval_data, final_generation, inference_results)
                # evaluation
                acc = self.EvalData.eval_acc(inference_results)
                print(f'{self.task} Accuracy in {idx} turn: {acc}')
                EM = self.EvalData.eval_exact_match(inference_results)
                print(f'{self.task} Exact match in {idx} turn: {EM}')
                f1_score = self.EvalData.eval_f1_score(inference_results)
                print(f'{self.task} F1 score in {idx} turn: {f1_score}')
            # end of for loop
            self.EvalData.save_result(inference_results)  # save inference results
            acc = self.EvalData.eval_acc(inference_results)
            print(f'{self.task} Accuracy: {acc}')
            EM = self.EvalData.eval_exact_match(inference_results)
            print(f'{self.task} Exact match in {idx} turn: {EM}')
            f1_score = self.EvalData.eval_f1_score(inference_results)
            print(f'{self.task} F1 score in {idx} turn: {f1_score}')
            eval_result = {'Accuracy':acc, 'Exact match': EM, 'F1 score':f1_score}
            return eval_result

    def dic2Example(self,dict_dataset:list[dict] )->list[Example]:
        '''
        The process of define dsp model need train dataset for demo whoes struction musk be list[Example]
        '''
        Example_dataset = []
        for sample_dict in dict_dataset:
            temp = {
                self.EvalData.inputStruction.question: sample_dict[self.EvalData.inputStruction.question],
                self.EvalData.inputStruction.answer: sample_dict[self.EvalData.inputStruction.answer]
            }
            Example_dataset.append(Example(temp).with_inputs(self.EvalData.inputStruction.question))
        return Example_dataset[:5]

    def load_llm(self):
        try:
            if self.model_mode == "HFModel":
                model = dspy.HFModel(model=self.llm_path) # dsp do not have the parameter of dtype, manual transfer fp32->fp16
                if self.dtype == 'half' or 'float16':
                    model = model.half()
                return model
            elif self.model_mode == "OpenAI":
                return dspy.OpenAI(model=self.llm_api, api_key=self.api_key, api_base=self.api_base)
            else:
                raise ValueError(f"Invalid model_mode: {self.model_mode}. Must be 'HFModel' or 'OpenAI'.")
        except ValueError as e:
            print(e)

    def get_instruction(self):
        return self.lm.inspect_history(n=1)
            
    def setup_Signature(self):
        if self.signature_retrieval:
            return dsp_utils.GenerateSearchQuery
        else:
            return dsp_utils.BasicQA

    def setup_generator(self):
        if self.inference_CoT:
            print(type(self.setup_Signature()))
            return dspy.ChainOfThought(self.setup_Signature(), temperature=self.temperature)
        else:
            return dspy.Predict(self.setup_Signature(),temperature=self.temperature)