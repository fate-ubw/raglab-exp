from typing import Optional
from tqdm import tqdm
import pdb
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag, ModeNotFoundError

class ItertiveRag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        self.init(args)

    def init(self,args):
        self.max_iteration = args.max_iteration

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
                question = eval_data[self.EvalData.inputStruction.question] 
                # infer
                final_answer, generation_track = self.infer(question)
                # record
                inference_results = self.EvalData.record_result(eval_data, final_answer, inference_results)
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
            print(f'{self.task} in {idx} turn: \n Accuracy: {acc} \n Exact match:{EM} \n F1 score: {f1_score}')
            eval_result = {'Accuracy':acc, 'Exact match': EM, 'F1 score':f1_score}
            return eval_result
        else:
            raise ModeNotFoundError("Mode must be interact or evaluation. Please provide a valid mode.")
    
    def infer(self, query:str)-> tuple[int,dict]:
        '''
        paper:
        source code: none
        '''
        generation_track = {}
        generation_track[0] = {'instruction': None, 'retrieval_input': query, 'passages':None, 'generation':None}
        for iter in range(self.max_iteration):
            retrieval_input = generation_track[iter]['retrieval_input']
            passages = self.retrieval.search(retrieval_input)
            collated_passages = self.collate_passages(passages)
            target_instruction = self.find_instruction('Iterative_rag-read', self.task)
            input = target_instruction.format_map({'passages': collated_passages, 'query': query})
            output = self.llm_inference(input)
            generation_track[iter]['instruction'] = input
            generation_track[iter]['cited passages'] = passages
            generation_track[iter]['final answer'] = output
            # save outputs as next iter retrieval inputs
            generation_track[iter+1] = {'instruction':None, 'retrieval_input': output, 'passages':None, 'generation':None}
        return output, generation_track