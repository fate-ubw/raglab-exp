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
        collection = Collection(path = collection_path) 
        try:
            # try reuse mode at first. If the index break,
            with Run().context(RunConfig(nranks = self.num_gpu, experiment = self.index_dbPath)):  # nranks specifies the number of GPUs to use.
                config = ColBERTConfig(doc_maxlen = self.doc_maxlen, nbits = self.nbits, kmeans_niters = 4) #
                indexer = Indexer(checkpoint = self.retriever_modelPath, config = config)
                indexer.index(name = index_name, collection = collection, overwrite='reuse') # set reuse mode
            with Run().context(RunConfig(experiment = self.index_dbPath)): 
                self.searcher = Searcher(index = index_name)
        except:
            with Run().context(RunConfig(nranks = self.num_gpu, experiment = self.index_dbPath)):  # nranks specifies the number of GPUs to use.
                config = ColBERTConfig(doc_maxlen = self.doc_maxlen, nbits = self.nbits, kmeans_niters = 4) #
                indexer = Indexer(checkpoint = self.retriever_modelPath, config = config)
                indexer.index(name = index_name, collection = collection, overwrite=True) # set reuse mode
            with Run().context(RunConfig(experiment = self.index_dbPath)): 
                self.searcher = Searcher(index = index_name)
        
    def search(self, query) -> dict[int,dict]:
        ids = self.searcher.search(query, k = self.n_docs)
        passages = {}
        for passage_id, passage_rank, passage_score in zip(*ids):
            print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {self.searcher.collection[passage_id]}")
            passages[passage_rank] = {'content': self.searcher.collection[passage_id], 'score':passage_score}
        return passages
