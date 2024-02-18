
import os
import sys
sys.path.insert(0, '../')
import pdb
import faiss
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
import pudb
if __name__=='__main__':
    dataroot = '/home/zxw/raglab-exp/data/retrieval/colbertv2.0_passages/lotte'
    dataset = 'lifestyle'
    datasplit = 'dev'
    queries = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
    collection = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')

    queries = Queries(path=queries) # Q: 这个类是干啥的呢？？？？，
    collection = Collection(path=collection)

    f'Loaded {len(queries)} queries and {len(collection):,} passages'
    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300   # truncate passages at 300 tokens 这个参数是 config 会设定的参数

    checkpoint = '/home/zxw/rag/model/colbertv2.0'
    index_name = f'{dataset}.{datasplit}.{nbits}bits'
    # Q：这个框架是否实现了自查有没有 idnex 函数，
    with Run().context(RunConfig(nranks=8, experiment='/home/zxw/raglab-exp/data/retrieval/colbertv2.0_embedding')):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4)        
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True) #assert overwrite in [True, False, 'reuse', 'resume', "force_silent_overwrite"] 如果没有的话，应该会重新创建
    indexer.get_index() # You can get the absolute path of the index, if needed.
    pu.db
    with Run().context(RunConfig(experiment='/home/zxw/raglab-exp/data/retrieval/colbertv2.0_embedding')):
        searcher = Searcher(index=index_name) 
    # (Pdb) searcher.config
    # ColBERTConfig(query_token_id='[unused0]', doc_token_id='[unused1]', query_token='[Q]', doc_token='[D]', ncells=1, centroid_score_threshold=0.5, ndocs=256, load_index_with_mmap=False, index_path=None, index_bsize=64, nbits=2, kmeans_niters=20, resume=False, similarity='cosine', bsize=64, accumsteps=1, lr=1e-05, maxsteps=400000, save_every=None, warmup=20000, warmup_bert=None, relu=False, nway=64, use_ib_negatives=True, reranker=False, distillation_alpha=1.0, ignore_scores=False, model_name=None, query_maxlen=32, attend_to_mask_tokens=False, interaction='colbert', dim=128, doc_maxlen=300, mask_punctuation=True, checkpoint='/home/wyd/model/colbertv2.0', triples='/future/u/okhattab/root/unit/experiments/2021.10/downstream.distillation.round2.2_score/round2.nway6.cosine.ib/examples.64.json', collection=<colbert.data.collection.Collection object at 0x7f1b86b75460>, queries='/future/u/okhattab/data/MSMARCO/queries.train.tsv', index_name='lifestyle.dev.2bits', overwrite=False, root='/home/wyd/ColBERT/experiments', experiment='notebook', index_root=None, name='2024-02/08/07.16.56', rank=0, nranks=1, amp=True, gpus=2, avoid_fork_if_possible=False)
    pdb.set_trace()
    query = queries[37]   # or supply your own query
    # 这个 query 拿到的是 str 类型的数据，也就是说retrieval 可以直接接受 text 类型的数据，这样就方便多了
    print(f"#> {query}")
    pdb.set_trace()
    # Find the top-3 passages for this query
    results = searcher.search(query, k=3)

    # Print out the top-k retrieved passages
    for passage_id, passage_rank, passage_score in zip(*results):
        print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")
    # (Pdb) searcher.collection[0]
    # "In my experience rabbits are very easy to housebreak. They like to pee and poop in the same place every time, so in most cases all you have to do is put a little bit of their waste in the litter box and they will happily use the litter box. It is very important that if they go somewhere else, miss the edge or kick waste out of the box that you clean it up well and immediately as otherwise those spots will become existing places to pee and poop. When you clean the box, save a little bit of waste and put it in the cleaned box so it smells right to them. For a more foolproof method, you can get a piece of wood soaked with their urine and put that in the box along with droppings or cage them so that they are only in their litter box for a week. Generally, if I try the first method and find that they are not using only the box on the first day, I go for the litter box only for a week method. The wood block works well if you are moving from a hutch outdoors to a litter box indoors. If you have an indoor cage, you can use the cage itself as the litter box (or attach a litter box to the section of the cage the rabbit has used for waste.) Be sure to use clay or newsprint litter as the other types aren't necessarily good for rabbits. Wood litter is okay if you are sure it isn't fir. The most important thing is to clean anywhere they have an accident. H
    # 这个就是拿到的 passages
