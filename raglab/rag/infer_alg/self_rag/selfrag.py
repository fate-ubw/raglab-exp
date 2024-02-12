from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
from raglab.rag.infer_alg.self_rag.utils import load_special_tokens

class SelfRag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        self.init(args)
        #super相当于NaiveRag 类的实例

    def init(self, args):
        self.download_dir = args.download_dir
        self.world_size = args.world_size
        self.dtype = args.dtype
        self.threshold = args.threshold
        self.use_seqscore = args.use_seqscore
        self.use_groundness = args.use_groundness
        self.use_utility = args.use_utility
        self.beam_width = args.beam_width
        self.max_depth = args.max_depth
        self.w_rel = args.w_rel
        self.w_sup = args.w_sup
        self.w_use = args.w_use
        self.retrieval_mode = args.retrieval_mode

    def inference(self, query=None, mode='interact', task=None):
        # mode 肯定是有的，如果有 mode 还得封装一层
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode: #
            pass   
        elif 'evaluation' == mode:
            if 'PopQA' == self.task:
                pass

    def load_llm(self): # self rag 的代码必须得使用 vllm 来进行生成，因为这样才能比较方便的得到log prob
        llm = None
        tokenizer = None
        llm = LLM(model=self.llm_path) 
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        tokenizer = AutoTokenizer.from_pretrained(gpt, padding_side="left")
        return llm, tokenizer    
    def get_prompt(self, passages, query):# self rag好想不需要 get prompt，因为这个框架和其它所有的都不太一样
        return super().get_prompt(passages, query)

    def generation(self, prompt, evidences, model, max_new_tokens = 300,
                    ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                    use_seqscore=False, threshold=0.5,
                    w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):# 这个里面的参数最好自己来实现
        # self.para是没有办法直接当做 default 值的因为只有定义的时候才能调用 self，在编译的时候还没有 self
        #load special token
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        self.tokenizer, use_grounding=self.use_groundness, use_utility=self.use_utility)
        

        pass
