from raglab.rag.infer_alg.naive_rag import NaiveRag

class SelfRag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        #super相当于NaiveRag 类的实例
    
    def inference(self, query=None, mode='interact', task=None):
    
        pass
    
         
    def get_prompt(self, passages, query):
        return super().get_prompt(passages, query)
