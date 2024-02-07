class NiaveRag:
    def __init__(self, args):
        
        # common args
        self.top_k = args.top_k
        self.llm_path = args.llm_path # __init__ 只进行参数的传递，尤其是传递路径什么的
        self.documents_path = args.documents_path # 也就是说这里直接接受的是处理好的数据
        self.eval_datapath = args.eval_datapath
        self.db_path = args.db_path

        # load database
        self.database = self.load_dataset()

        # load eval dataset
        self.eval_dataset = self.load_evaldataset()

        # load llm
        self.llm = self.load_llm()
        # load retriever
        self.retrieval = self.load_retriever() 

    def init(self):
        # 
        raise NotImplementedError

    def inference(self, query):
        # 这部分好像不需要 embedding，因为直接可以调用 self.retrieval(query) 然后就可以直接得到结果
        passages = self.retrieval(query)
        inputs = preprocess(passages, query) # 这部分就需要设计一个 prompt 合并 query和 passages
        outputs = self.llm(inputs)
        response = postporcess(outputs)
        # 
        return response
    def set_prompt(self): #prompt 如果子类不进行继承，那么就使用 naive_rag 的 prompt
        template = '''
            【任务描述】
            请根据用户输入的上下文回答问题，并遵守回答要求。
            【背景知识】
            {{context}}

            【回答要求】
            - 你需要严格根据背景知识的内容回答，禁止根据常识和已知信息回答问题。
            - 对于不知道的信息，直接回答“未找到相关答案”
            -----------
            {question}
            '''
        return prompt
        
    