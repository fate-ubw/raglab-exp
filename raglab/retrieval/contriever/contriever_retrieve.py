import os
import argparse
from tqdm import tqdm
import time
import glob
import json
import jsonlines
import pickle
from pathlib import Path
import numpy as np
import torch
import transformers

from raglab.retrieval.contriever.src.index import Indexer
from raglab.retrieval.contriever.src.contriever import load_retriever
from raglab.retrieval.contriever.src.slurm import init_distributed_mode
from raglab.retrieval.contriever.src.data import load_passages
from raglab.retrieval.contriever.src.evaluation import calculate_matches
from raglab.retrieval.contriever.src.normalize_text import normalize
from raglab.retrieval.contriever.utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from raglab.retrieval.retrieve import Retrieve
import pudb

os.environ["TOKENIZERS_PARALLELISM"] = "true"
class ContrieverRrtieve(Retrieve):
    def __init__(self, args):
        self.args = args
        self.index_dbPath = args.index_dbPath
        self.text_dbPath = args.text_dbPath
        self.retriever_modelPath = args.retriever_modelPath
        self.projection_size = args.projection_size
        self.indexing_batch_size = args.indexing_batch_size # 这个 batch 好想是可以统一起来的
        self.n_subquantizers = args.n_subquantizers 
        self.n_bits = args.n_bits 
        self.n_docs = args.n_docs

    def setup_retrieve(self):
        # 好 setup 已经完成了，可以实验一下了
        init_distributed_mode(self.args) #这个需要传递所有的args
        print(f"Loading model from: {self.retriever_modelPath}")
        self.model, self.tokenizer, _ = load_retriever(self.retriever_modelPath)
        self.model.eval()
        self.model = self.model.cuda()
        self.model = self.model.half() # (32-bit) floating-point -> (16-bit) floating-point  
        self.index = Indexer(self.projection_size, self.n_subquantizers, self.n_bits) # 貌似self rag 里面也存在这个东西
        input_paths = glob.glob(self.index_dbPath) # path of embedding
        input_paths = sorted(input_paths)
        # embeddings_dir = os.path.dirname(input_paths[0]) 
        # index_path = os.path.join(embeddings_dir, "index.faiss")
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        self.index_encoded_data(self.index, input_paths, self.indexing_batch_size) # load all embedding files（index_encoded_data）& （add_embeddings）
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        print("loading passages") 
        self.passages = load_passages(self.text_dbPath) # return list of passages
        self.passage_id_map = {x["id"]: x for x in tqdm(self.passages)} # 将passages定义为{'n':conent} 
        print("passages have been loaded") 

    def search(self, query): # 对外的时候是不需要topk 参数，这个参数在 yaml 文件里面定义，大部分情况top-k 是固定
        passages = {}
        #这个返回的维度得把握一下
        questions_embedding = self.embed_queries(self.args, [query]) # 这里直接就是整个数据集
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, self.n_docs)  # 输入是什么样子的，这个得看一下，之后就是
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.") # search_knn 是用来计算 score 的，其实这个也就够用了
        passages = self.add_passages(self.passage_id_map, top_ids_and_scores)[:self.n_docs] #这里面直接去 top-k 应该就可以了吧但是为什么还要使用 top-n
        return passages #这里面缺少 score，和 rank 返回的只有一个 list of passages

    def index_encoded_data(self, index, embedding_files, indexing_batch_size): 
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files): # 这个为什么拿不出来呢？？？？？
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin) # 其实最终的数据是 idx 和对应的embedding

            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids) #
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        print("Data indexing completed.")

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0]) # 首先是规定了 embedding idx 的最大值，所有 idx 都不会超过这个值
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx] # 规定load 的全部数据
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd) # 其实这个是最重要的 将全部的idx 和 embedding都加载到 index 
        return embeddings, ids

    def embed_queries(self, args, queries): # 看来这个函数的输入是不对的，
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries): #看来也是可以处理多个 query 的但是无法实现并行
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:  # 专门处理\n 这样的字符串的函数
                    q = normalize(q)
                batch_question.append(q)

                if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength, # 默认最大长度是 512 ？如果是
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())
                    batch_question = []
        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")
        return embeddings.numpy() #这个的维度是多少呢
    
    def add_passages(self, passages, top_passages_and_scores):
        # add passages to original data
        # 因为 passages 使用的是
        docs = [passages[doc_id] for doc_id in top_passages_and_scores[0][0]]
        return docs