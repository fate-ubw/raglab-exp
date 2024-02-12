import argparse
import pdb
import pudb

from raglab.rag.infer_alg.naive_rag import NaiveRag
from utils import over_write_args_from_file
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type = int, default = 1, help = 'the number of gpu')
    parser.add_argument('--output_dir', type = str, help = 'the output dir of evaluation')
    parser.add_argument('--mode', type = str, default = 'interact', choices = ['interact', 'evaluation'], help = 'different mode of ingerence')
    parser.add_argument("--llm_path", type = str, help = 'path to llm')
    # retrieval config
    parser.add_argument('--retrieval_name', type = str, default = 'colbert', choices = ['colbert','contriever'],help = 'the name of retrieval model')
    parser.add_argument("--index_dbPath", type = str, help = 'path to index database. Index is index and embedding pairs')
    parser.add_argument('--text_dbPath', type = str, help='path to text database')
    parser.add_argument("--eval_datapath", type = str, help = 'path to eval dataset')
    parser.add_argument("--retriever_modelPath", type = str, help = 'path to colbert model')
    parser.add_argument('--generate_maxlength', type = int, default = 50, help = 'llm generate max length')
    parser.add_argument("--n_docs", type= int, default=10, help="Number of documents to retrieve per questions")
    parser.add_argument('--use_vllm', action = "store_true", help = 'llm generate max length')
    parser.add_argument('--doc_maxlen', type = int, default = 300, help = 'the doc max len decided by the wikidata format, here we set 300')
    parser.add_argument('--nbits', type = int, default = 2, help = 'encode each dimension with n bits')
    
    # contrieval config
    parser.add_argument('--projection_size', type = int, default=768, help = 'size of embedding') # righ
    parser.add_argument('--n_subquantizers', type = int, default=0, help="Number of subquantizer used for vector quantization, if 0 flat index is used")
    parser.add_argument('--n_bits', type = int, default = 8, help="Number of bits per subquantizer")
    parser.add_argument("--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed")   
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")# 目前不需要设置这个参数
    parser.add_argument("--normalize_text", action="store_true", help="normalize text") #调用 retrieval 不需要使用
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    
    # config file
    parser.add_argument('--config',type = str, default = "")
    args = parser.parse_args()
    over_write_args_from_file(args, args.config) #
    return args

if __name__=='__main__':
    args = get_config()
    rag = NaiveRag(args)# so we can edit the file in vim 
    pu.db
    eval_result = rag.inference(mode = 'evaluation', task = "PopQA") #参数在定义 Naiverag 的时候就传进去了，这部分不需要担心
    print(eval_result)
