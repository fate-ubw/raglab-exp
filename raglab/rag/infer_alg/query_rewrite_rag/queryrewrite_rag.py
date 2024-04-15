from typing import Optional, Any
from tqdm import tqdm
import pdb
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag

class QueryRewrite_rag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)

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

