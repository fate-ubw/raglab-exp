import os
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import faiss
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from utils import load_evaldataset, save_inference_result
import pdb
from tqdm import tqdm
import json
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

    def inference(self, query = None, mode = 'interact'):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            passages = self.search(query) # 这里我感觉可以构造一个 dic()
            # passages: dict of dict
            inputs = self.get_prompt(passages, query) # 这部分就需要设计一个 prompt 合并 query和 passages
            outputs = self.llm_inference(inputs) # 
            response = self.postporcess(outputs) # 不同的任务可能需要使用不同的文本处理方法
            return response
        elif 'evaluation' == mode: 
            self.eval_dataset = load_evaldataset(self.eval_datapath) # 不同数据集合使用不同的 load 方法不一样的，input output 不一样
            #把握 input 和 output
            inference_result = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                question = eval_data["question"] # 这个参数是和具体数据相关的，这个 key 选什么也没有什么办法，到时候放到 dataset 里面
                passages = self.search(question)
                inputs = self.get_prompt(passages, question)
                outputs = self.llm_inference(inputs)
                response = self.postporcess(outputs)
                eval_data["generation"] = response 
                inference_result.append(eval_data)
            save_inference_result(inference_result, self.output_dir, self.llm_path, self.eval_datapath)
            # 这个函数应该是 PopQa.save()这样就比较合理了，不需要全部的
            # PopQA.evaluate() 直接进行评价，这部分还是直接实
            pdb.set_trace()
            # 优先解决args 的问题还是有

            print('start evaluation!')
            eval_result = eval_PopQA(args) #
            print(eval_result)
            # evaluation 在dataset文件夹下面
            # - popQA.py 
            # 每一个 dataset 写一个.py文件：load 
            #  可以设计一个 baseclass 来实现 multi-choice
            
            return eval_result


    def load_llm(self):
        # load tokenizer and llm
        # todo: vllm的参数设置也必须统一起来
        llm = None
        tokenizer = None
        if self.use_vllm:
            llm = LLM(model=self.llm_path) # 成功加载
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
    
    def get_prompt(self, passages, query): #prompt 如果子类不进行继承，那么就使用 naive_rag 的 prompt
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
    
    def postporcess(self, samples): # naive rag 不需要对生成的结果进行更多的操作，但是根据不同的任务需要对 special token 进行处理的
        
        processed_samples = samples
        return processed_samples

    def llm_inference(self, inputs): # 内置调用 llm 的算法 
        if self.use_vllm:
            output = self.llm.generate(inputs, self.sampling_params)
            output_text = output[0].outputs[0].text
        else:
            input_ids = self.tokenizer.encode(inputs, return_tensors="pt")
            output_ids = self.llm.generate(input_ids, do_sample = False, max_length = self.generate_maxlength)
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens = False)
        #到时候可以写一个 vllm 的开关，但是 load 的时候就需要决定使用哪种算法
        if '<\s>' in output_text: # 因为
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