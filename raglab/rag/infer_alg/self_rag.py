from raglab.rag.infer_alg.naive_rag import NaiveRag

class SelfRag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        #super相当于NaiveRag 类的实例

    def init(self):#其余的参数其实可以写到这里面来，传递一个 args 即可相当于这些参数都是 self rag 自己独有的
        pass
    
    def inference(self, query=None, mode='interact', task=None):
        # mode 肯定是有的，如果有 mode 还得封装一层
        assert mode in ['interact', 'evaluation']
        assert task in ['PopQA']
        if 'interact' == mode: #
            pass   
        elif 'evaluation' == mode:
            if 'PopQA' == task:
                pass
    
    def get_prompt(self, passages, query):# self rag好想不需要 get prompt，因为这个框架和其它所有的都不太一样
        return super().get_prompt(passages, query)


