import faiss
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig

from raglab.retrieval.retrieve import Retrieve

class ColbertRetrieve(Retrieve):
    def __init__(self, args):
        self.index_dbPath = args.index_dbPath
        self.text_dbPath = args.text_dbPath
        self.retriever_modelPath = args.retriever_modelPath
        
        self.nbits = args.nbits
        self.num_gpu = args.num_gpu
        self.doc_maxlen = args.doc_maxlen
        self.n_docs = args.n_docs
        
    def setup_retrieve(self):
        dataset = 'lifestyle'
        datasplit = 'dev'
        index_name = f'{dataset}.{datasplit}.{self.nbits}bits' # TODO 知识库的名字需要后期再想办法参数化
        collection_path = self.text_dbPath
        collection = Collection(path = collection_path) # 尝试一下能不能直接

        with Run().context(RunConfig(nranks = self.num_gpu, experiment = self.index_dbPath)):  # nranks specifies the number of GPUs to use.
            config = ColBERTConfig(doc_maxlen = self.doc_maxlen, nbits = self.nbits, kmeans_niters = 4) #
            indexer = Indexer(checkpoint = self.retriever_path, config = config)
            indexer.index(name = index_name, collection = collection, overwrite='reuse') # here we set reuse mode

        with Run().context(RunConfig(experiment = self.index_dbPath)): 
            self.searcher = Searcher(index = index_name)
        # 这里才完成 self.searcher 的定义下面再进行 seearch 的封装即可
    
    def search(self, query):
        ids = self.searcher.search(query, k = self.n_docs)
        passages = {}
        for passage_id, passage_rank, passage_score in zip(*ids): # 这里面的*是用来解耦元素的，将整个 list 全部变成一个单独的个体
            print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {self.searcher.collection[passage_id]}")
            passages[passage_rank] = {'content': self.searcher.collection[passage_id], 'score':passage_score}
        pu.db
        return passages
