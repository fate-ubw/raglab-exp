
import os
import sys
sys.path.insert(0, '../')
import pdb
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
import pudb
if __name__=='__main__':
    checkpoint = "/workspace/raglab-exp/model/colbertv2.0"
    index_dbPath = '/workspace/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2023'
    dataset = os.path.basename(index_dbPath)
    collection = '/workspace/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2023/enwiki-20230401.tsv'
    collection = Collection(path=collection)
    f'Loaded {len(collection):,} passages'
    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300  
    index_name = dataset 
    with Run().context(RunConfig(nranks=4, experiment=index_dbPath)):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4)        
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True) #assert overwrite in [True, False, 'reuse', 'resume', "force_silent_overwrite"]
    indexer.get_index() # You can get the absolute path of the index, if needed.
    with Run().context(RunConfig(experiment=index_dbPath)):
        searcher = Searcher(index=index_name)
    # (Pdb) searcher.config
    # ColBERTConfig(query_token_id='[unused0]', doc_token_id='[unused1]', query_token='[Q]', doc_token='[D]', ncells=1, centroid_score_threshold=0.5, ndocs=256, load_index_with_mmap=False, index_path=None, index_bsize=64, nbits=2, kmeans_niters=20, resume=False, similarity='cosine', bsize=64, accumsteps=1, lr=1e-05, maxsteps=400000, save_every=None, warmup=20000, warmup_bert=None, relu=False, nway=64, use_ib_negatives=True, reranker=False, distillation_alpha=1.0, ignore_scores=False, model_name=None, query_maxlen=32, attend_to_mask_tokens=False, interaction='colbert', dim=128, doc_maxlen=300, mask_punctuation=True, checkpoint='/home/wyd/model/colbertv2.0', triples='/future/u/okhattab/root/unit/experiments/2021.10/downstream.distillation.round2.2_score/round2.nway6.cosine.ib/examples.64.json', collection=<colbert.data.collection.Collection object at 0x7f1b86b75460>, queries='/future/u/okhattab/data/MSMARCO/queries.train.tsv', index_name='lifestyle.dev.2bits', overwrite=False, root='/home/wyd/ColBERT/experiments', experiment='notebook', index_root=None, name='2024-02/08/07.16.56', rank=0, nranks=1, amp=True, gpus=2, avoid_fork_if_possible=False)
    # text of colbert search
    query = 'who is Aaron?'   
    print(f"#> {query}")
    results = searcher.search(query, k=3)
    for passage_id, passage_rank, passage_score in zip(*results):
        print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")