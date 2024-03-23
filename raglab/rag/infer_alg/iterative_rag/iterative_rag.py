from typing import Optional, Any
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pdb
import pudb
import re
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag


class ItertiveRag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        self.init(args)

    def init(self,args):
        self.max_iteration = args.max_iteration

    def inference(self, query: Optional[str] = None, mode = 'interact'):# mode 不会冲突因为这个mode 是函数内在的 mode
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            generation_track = {}
            generation_track[0] = {'instruction': None, 'retrieval_input': query, 'passages':None, 'generation':None}
            for iter in range(self.max_iteration):
                retrieval_input = generation_track[iter]['retrieval_input']
                passages = self.retrieval.search(retrieval_input)
                collated_passages = self.collate_passages(passages)
                target_instruction = self.find_instruction('Iterative_rag-read', self.task)
                inputs = target_instruction.format_map({'passages': collated_passages, 'query': query})
                outputs = self.llm_inference(inputs)
                generation_track[iter]['instruction'] = inputs
                generation_track[iter]['passages'] = passages
                generation_track[iter]['generation'] = outputs
                # save outputs as next iter retrieval inputs
                generation_track[iter+1] = {'instruction':None, 'retrieval_input': outputs, 'passages':None, 'generation':None}
            citation_passages = passages
            return outputs, citation_passages, generation_track # 所有 interact 都使用这 3 个标准的回答，但是selfrag 使用的是 beam search 还是需要注意一下的
        # end of interact mode
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset()
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                question = eval_data[self.EvalData.inputStruction.question] 
                generation_track = {}
                generation_track[0] = {'instruction':None, 'retrieval_input': question,'passages':None, 'generation':None}
                for iter in range(self.max_iteration):
                    # iterative retrieval
                    retrieval_input = generation_track[iter]['retrieval_input']
                    passages = self.retrieval.search(retrieval_input)
                    collated_passages = self.collate_passages(passages)
                    target_instruction = self.find_instruction('Iterative_rag-read', self.task)
                    inputs = target_instruction.format_map({'passages': collated_passages, 'query': question})
                # read
                    outputs = self.llm_inference(inputs)
                    generation_track[iter]['instruction'] = inputs
                    generation_track[iter]['passages'] = passages
                    generation_track[iter]['generation'] = outputs
                    # save outputs as next iter retrieval inputs
                    generation_track[iter+1] = {'instruction':None, 'retrieval_input': outputs, 'passages':None, 'generation':None}
                    # 每个算法要保存的东西是不一样的，但是针对这样更加复杂的算法是否要
                inference_results = self.EvalData.record_result(eval_data, outputs, inference_results)
                eval_result = self.EvalData.eval_acc(inference_results)
                print(f'{self.task} Accuracy in {idx} turn: {eval_result}')
            # end of for loop
            self.EvalData.save_result(inference_results)
            eval_result = self.EvalData.eval_acc(inference_results) 
            print(f'{self.task} Accuracy: {eval_result}')
            return eval_result