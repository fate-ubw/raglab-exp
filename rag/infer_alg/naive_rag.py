import argparse

class NiaveRag:
    def __init__(self, args):
        
        # common args
        self.n_docs = args.n_docs
        self.llm_path = args.llm_path # __init__ 只进行参数的传递，尤其是传递路径什么的
        self.retriever_path = args.retriever_path # 也就是说这里直接接受的是处理好的数据

        self.eval_datapath = args.eval_datapath
        self.db_path = args.db_path

        # load database
        self.database = self.load_database() # 这部分我感觉可以合并到 setup_retrieval 部分

        # load eval dataset
        self.eval_dataset = self.load_evaldataset() # 

        # load llm
        self.llm = self.load_llm()
        # load retriever
        self.retrieval = self.setup_retrieval() 

    def init(self):
        # 
        raise NotImplementedError

    def inference(self, query):
        # 这部分好像不需要 embedding，因为直接可以调用 self.retrieval(query) 然后就可以直接得到结果
        passages = self.retrieval(query) 
        inputs = get_prompt(passages, query) # 这部分就需要设计一个 prompt 合并 query和 passages
        outputs = self.llm(inputs)
        response = postporcess(outputs) # 不同的任务可能需要使用不同的文本处理方法，
        # 
        return response

    def load_database(self):
        pass

    def load_evaldataset(self):
        pass

    def load_llm(self):
        # 这里直接就是直接使用 huggingface 那一套 load 进来然后返回模型应该就可以了
        pass

    def setup_retrieval(self):
        pass

    def get_prompt(self): #prompt 如果子类不进行继承，那么就使用 naive_rag 的 prompt
        template = '''
            [任务描述]
            请根据用户输入的上下文回答问题，并遵守回答要求。
            [背景知识]
            {{context}}

            [回答要求]
            - 你需要严格根据背景知识的内容回答，禁止根据常识和已知信息回答问题。
            - 对于不知道的信息，直接回答“未找到相关答案”
            -----------
            [问题]
            {question}
            '''
        return prompt
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type=int, default=10, help="Number of documents to retrieve per questions")
    parser.add_argument("--llm_path", type = str, help = 'path to llm')
    parser.add_argument("--db_path", type = str, help = 'path to preprocessed databset with index')
    parser.add_argument("--eval_datapath", type = str, help = 'path to eval dataset')
    parser.add_argument("--retriever_path", type = str, help = 'path to colbert model')
    args = parser.parse_args()
    rag = NiaveRag(args)
    print(rag)
