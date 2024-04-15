from typing import Optional
from tqdm import tqdm
import pdb
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag

class ItertiveRag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        self.init(args)

    def init(self,args):
        self.max_iteration = args.max_iteration

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