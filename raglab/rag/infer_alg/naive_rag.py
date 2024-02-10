import os
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import faiss
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
import pdb
from tqdm import tqdm
import json

from raglab.dataset.PopQA import PopQA # load popqa class
from raglab.rag.infer_alg.utils import load_evaldataset, save_inference_result
class NaiveRag:
    def __init__(self, args):
        
        # common args
        self.llm_path = args.llm_path # __init__ 只进行参数的传递，尤其是传递路径什么的
        self.generate_maxlength = args.generate_maxlength
        self.use_vllm = args.use_vllm
        self.num_gpu = args.num_gpu

        self.eval_datapath = args.eval_datapath
        self.output_dir = args.output_dir

        # retrieval args
        self.n_docs = args.n_docs
        self.nbits = args.nbits
        self.doc_maxlen = args.doc_maxlen # 这个后期其实可以固定下来
        self.retriever_path = args.retriever_path # 也就是说这里直接接受的是处理好的数据
        self.db_path = args.db_path

        # load model and database
        self.llm, self.tokenizer = self.load_llm() # 传一个args
        self.retrieval = self.setup_retrieval()

    def init(self):
        # 
        raise NotImplementedError

    def inference(self, query = None, mode = 'interact', task = None):
        assert mode in ['interact', 'evaluation']
        assert task in ['PopQA']
        if 'interact' == mode:
            passages = self.search(query)
            # passages: dict of dict
            inputs = self.get_prompt(passages, query) 
            outputs = self.llm_inference(inputs) # 
            response = self.postporcess(outputs) 
            return response
        elif 'evaluation' == mode:
            if 'PopQA' == task:
                popqa =  PopQA(self.output_dir, self.llm_path, self.eval_datapath) 
                self.eval_dataset = popqa.load_dataset() #
                
                inference_results = []
                for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                    question = eval_data["question"] # 这个参数是和具体数据相关的，这个 key 选什么也没有什么办法，到时候放到 dataset 里面
                    passages = self.search(question)
                    inputs = self.get_prompt(passages, question)
                    outputs = self.llm_inference(inputs)
                    eval_data["generation"] = outputs 
                    inference_results.append(eval_data)
                
                popqa.save_result(inference_results, self.output_dir)
                eval_result = popqa.eval_acc(inference_results) 
                print(f'PopQA accuracy: {eval_result}')
            return eval_result 

    def load_llm(self):
        llm = None
        tokenizer = None
        if self.use_vllm:
            llm = LLM(model=self.llm_path) 
            self.sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_path, skip_special_tokens=False) #
            llm = AutoModelForCausalLM.from_pretrained(self.llm_path)
        return llm, tokenizer
        
    def setup_retrieval(self):
        # TODO: idnex_name 后续得想的简单点，不然参数太多了，直接给一个 wiki 的 encode 就完了
        dataroot = '/home/wyd/data/4-colbert/lotte'
        dataset = 'lifestyle'
        datasplit = 'dev'
        index_name = f'{dataset}.{datasplit}.{self.nbits}bits'
        collection_path = os.path.join(dataroot, dataset, datasplit, 'collection.tsv') # 外挂数据的路径还是需要想办法解决一下
        collection = Collection(path = collection_path)
        with Run().context(RunConfig(nranks = self.num_gpu, experiment = self.db_path)):  # nranks specifies the number of GPUs to use.
            config = ColBERTConfig(doc_maxlen = self.doc_maxlen, nbits = self.nbits, kmeans_niters = 4) #
            indexer = Indexer(checkpoint = self.retriever_path, config=config)
            indexer.index(name = index_name, collection = collection, overwrite='reuse') # here we set reuse mode
        
        with Run().context(RunConfig(experiment = self.db_path)):
            searcher = Searcher(index = index_name)
        return searcher
    
    def get_prompt(self, passages, query): 
        # passages is dict type 
        # 不对不同任务有不同的 prompt 这部分直接就
        collater = ''
        for rank_id, tmp in passages.items():
            collater += f'Passages{rank_id}: ' + tmp['content'] +'\n'  # 这个拿回来之后             
        prompt = f'''
                [Task]
                Please answer the question based on the user's input context and comply with the answering requirements.
                [Background Knowledge]
                {collater}
                [Answering Requirements]
                - You need to strictly answer based on the content of the background knowledge, and it is forbidden to answer questions based on common sense and known information.
                - For information that is not known, simply answer "No relevant answer found"
                [Question]
                {query}
                '''
        # 感觉这块可以添加不同任务的 instruction 因为不同任务使用的instruction 是不一样的
        return prompt

    def llm_inference(self, inputs): 
        if self.use_vllm:
            output = self.llm.generate(inputs, self.sampling_params)
            output_text = output[0].outputs[0].text
        else:
            input_ids = self.tokenizer.encode(inputs, return_tensors="pt")
            output_ids = self.llm.generate(input_ids, do_sample = False, max_length = self.generate_maxlength)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens = False)
        if '<\s>' in output_text: 
            return output_text.replace("<s> " + inputs, "").replace("</s>", "").strip()
        else:
            return output_text.replace("<s> " + inputs, "").strip()
    
    def search(self, query):
        ids = self.retrieval.search(query, k = self.n_docs)
        passages = {}
        for passage_id, passage_rank, passage_score in zip(*ids): # 这里面的*是用来解耦元素的，将整个 list 全部变成一个单独的个体
            print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {self.retrieval.collection[passage_id]}")
            passages[passage_rank] = {'content': self.retrieval.collection[passage_id], 'score':passage_score}
        return passages