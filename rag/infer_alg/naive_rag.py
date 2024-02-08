import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
class NiaveRag:
    def __init__(self, args):
        
        # common args
        self.n_docs = args.n_docs
        self.llm_path = args.llm_path # __init__ 只进行参数的传递，尤其是传递路径什么的
        self.generate_maxlength = args.generate_maxlength
        self.retriever_path = args.retriever_path # 也就是说这里直接接受的是处理好的数据
        self.use_vllm = args.use_vllm
        # 解码算法的模式选择 greedy 还有 nucleus 
        
        self.eval_datapath = args.eval_datapath
        self.db_path = args.db_path

        # load database
        self.database = self.load_database() # 这部分我感觉可以合并到 setup_retrieval 部分

        # load eval dataset
        self.eval_dataset = self.load_evaldataset() 

        # load llm  
        self.llm, self.tokenizer = self.load_llm() 

        # load retriever
        self.retrieval = self.setup_retrieval() 

    def init(self):
        # 
        raise NotImplementedError

    def inference(self, query):
        # 这部分好像不需要 embedding，因为直接可以调用 self.retrieval(query) 然后就可以直接得到结果
        passages = self.retrieval(query) 
        inputs = get_prompt(passages, query) # 这部分就需要设计一个 prompt 合并 query和 passages
        # 这部分必须是 str 才可以
        outputs = self.llm_inference(inputs) 
            # 这部分需要经过 3 个部分 encode -> generate -> decode
            # 所以还是集成起来比价合适
        response = postporcess(outputs) # 不同的任务可能需要使用不同的文本处理方法，
        # 
        return response

    def load_database(self):
        pass

    def load_evaldataset(self):
        pass

    def load_llm(self):
        # load tokenizer and llm
        # todo: vllm的参数设置也必须统一起来
        llm = None
        tokenizer = None
        if self.use_vllm:
            llm = LLM(model=self.llm_path) # 成功加载
            self.sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=50, logprobs=32000, skip_special_tokens = False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_path, skip_special_tokens=False) #
            llm = AutoModelForCausalLM.from_pretrained(self.llm_path)
        return llm, tokenizer
        
    def setup_retrieval(self):
        pass

    def get_prompt(self, passages, query): #prompt 如果子类不进行继承，那么就使用 naive_rag 的 prompt
        template = f'''
            [任务描述]
            请根据用户输入的上下文回答问题，并遵守回答要求。
            [背景知识]
            {context}
            [回答要求]
            - 你需要严格根据背景知识的内容回答，禁止根据常识和已知信息回答问题。
            - 对于不知道的信息，直接回答“未找到相关答案”
            [问题]
            {question}
            '''
        prompt = template.format(context = passages, question = query)
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