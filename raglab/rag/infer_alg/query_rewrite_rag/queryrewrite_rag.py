from typing import Optional, Any
from tqdm import tqdm
import pdb
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag, ModeNotFoundError

class QueryRewrite_rag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, query: Optional[str] = None, mode = 'interact'):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            final_response, generation_track = self.infer(query)
            return final_response, generation_track
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset() 
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                question = eval_data[self.EvalData.inputStruction.question]
                # infer
                final_response, generation_track = self.infer(question)
                inference_results = self.EvalData.record_result(eval_data, final_response, inference_results)
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
            

    def infer(self, query:str)->tuple[str, dict[str,Any]]:
        '''
        infer function of rrr
        paper:[https://arxiv.org/abs/2305.14283]
        source code: [https://github.com/xbmxb/RAG-query-rewriting/tree/main]
        '''
        # rewrite the query
        generation_track = {}
        instruction = self.find_instruction('query_rewrite_rag-rewrite', self.task)
        query_with_instruction = instruction.format_map({'query':query})
        rewrite_query = self._rewrite(query_with_instruction)
        generation_track['rewrite query'] = rewrite_query
        # retrieval
        passages = self.retrieval.search(rewrite_query)
        generation_track['cited passages'] = passages
        collated_passages = self.collate_passages(passages)
        instruction = self.find_instruction('query_rewrite_rag-read', self.task)
        query_with_instruction = instruction.format_map({'query':query, 'passages':collated_passages})
        # read
        output = self.llm_inference(query_with_instruction)
        generation_track['final answer'] = output
        return output, generation_track

    def _rewrite(self, query):
        rewrite_query = self.llm_inference(query)
        return rewrite_query

