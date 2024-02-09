import os
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import faiss
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from utils import load_jsonlines
import pdb
from tqdm import tqdm
class NaiveRag:
    def __init__(self, args):
        
        # common args
        self.mode = args.mode
        self.llm_path = args.llm_path # __init__ 只进行参数的传递，尤其是传递路径什么的
        self.generate_maxlength = args.generate_maxlength
        self.use_vllm = args.use_vllm
        self.num_gpu = args.num_gpu

        self.eval_datapath = args.eval_datapath

        # retrieval args
        self.n_docs = args.n_docs
        self.nbits = args.nbits
        self.doc_maxlen = args.doc_maxlen # 这个后期其实可以固定下来
        self.retriever_path = args.retriever_path # 也就是说这里直接接受的是处理好的数据
        self.db_path = args.db_path

        # load database & eval dataset
        if 'evaluation' == self.mode: # load evaluation dataset
            self.eval_dataset = self.load_evaldataset() #把握 input 和 output
            self.output_dir = args.output_dir
        elif 'interact' == self.mode: # 
            pass
        # load model and database
        self.llm, self.tokenizer = self.load_llm()
        self.retrieval = self.setup_retrieval()

    def init(self):
        # 
        raise NotImplementedError

    def inference(self, query):
        if 'interact' == self.mode:
            passages = self.search(query) # 这里我感觉可以构造一个 dic()
            # passages: dict of dict
            inputs = self.get_prompt(passages, query) # 这部分就需要设计一个 prompt 合并 query和 passages
            outputs = self.llm_inference(inputs) # 

            response = self.postporcess(outputs) # 不同的任务可能需要使用不同的文本处理方法
            return response
        elif 'evaluation' == self.mode: 
            self.eval_dataset 
            inference_result = []
            for idx, eval_data in tqdm(enumerate(self.eval_dataset)):
                qeustion = eval_data["question"]
                passages = self.search(question)
                inputs = self.get_prompt(passages, question)
                outputs = self.llm_inference(inputs)
                response = self.postporcess(outputs)
                eval_data["output"] = response
                inference_result.append(eval_data)
            # TODO 存储机制，这块得好好想想
            # 其实这块可以再进一步实现顺序的问题，也就是第一token 可以是 1,2,3,4 遍历所有文件，找到 basename 然后用-split 然后找到对第一个 token 排序，找到最大的，然后

            # check存储路径
            print('storing result....')
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            # 文件名称
            model_name = os.path.basename(self.llm_path)
            input_filename = os.path.basename(self.eval_datapath)
            eval_Dataname = os.path.splitext(input_filename)[0] #这个拿到的是dataset 的 name
            time = datetime.now().strftime('%m%d_%H%M') # time 
            output_name = f'infer_output-{eval_Dataname}-{model_name}-{time}.jsonl' #
            output_file = os.path.json(self.output_dir, output_name)
            # 存储文件
            with open(output_file, 'w') as outfile:
                for result in inference_result:
                    json.dump(result, outfile)
                    outfile.write('\n')
            print('success')
            eval_result = None
            return eval_result

    def load_evaldataset(self):
        if input_path.endswith(".json"):
            eval_dataset = json.load(open(self.eval_datapath))
        else:
            eval_dataset = load_jsonlines(self.eval_datapath) # 这一部分拿到的是一个 list of dict 
        # eval_dataset：type：list of dict
        return eval_dataset

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