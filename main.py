import argparse
import pdb
import pudb
import random
import torch
import numpy as np
from raglab.rag.infer_alg.self_rag_original import SelfRag_Original
from raglab.rag.infer_alg.self_rag_reproduction import SelfRag_Reproduction
from utils import over_write_args_from_file

def set_randomSeed(args):
    # random seed
    if args.use_seed == True:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default = 633, help='random  seed')
    parser.add_argument('--use_seed', action= 'store_true', help='this args will control all random seed of torch, numpy and pyhthon operation')
    parser.add_argument('--num_gpu', type = int, default = 1, help = 'the number of gpu')
    parser.add_argument('--output_dir', type = str, help = 'the output dir of evaluation')
    parser.add_argument('--task', type=str, choices=['PopQA','PubHealth','ArcChallenge', 'TriviaQA', 'ASQA', 'Factscore', 'HotpotQA', 'QReCC', 'SQuAD'], default=None, help='name of evaluation dataset')# task 参数影响 prompt 还有 format 
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

    # self rag config
    parser.add_argument('--download_dir', type=str, default=".cache",help="specify vllm model download dir")
    parser.add_argument("--world_size",  type=int, default=1,help="world size to use multiple GPUs.")
    parser.add_argument("--dtype", type=str, default= "half", help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
        # Decoding hyperparams
    parser.add_argument('--threshold', type=float, default=None, help="Adaptive threshold.")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true", help="use ground score")
    parser.add_argument("--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width", type=int, default=2, help="beam search width")
    parser.add_argument("--max_depth",type=int, default=2, help="tree depth width")
    parser.add_argument("--w_rel", type=float, default=1.0, help="reward weight for document relevance")
    parser.add_argument("--w_sup", type=float, default=1.0, help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use", type=float, default=1.0,help="reward weight for overall completeness / utility.")
    parser.add_argument('--retrieval_mode', type=str, help="mode to control retrieval.", default="no_retrieval", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieval']) 
    parser.add_argument('--show_specialtokens', action="store_true", help='show special tokens or remove all special tokens in outputs')
    parser.add_argument('--realtime_retrieval', action='store_true', help='self rag can use local passages(only)') # this setting ami to reproduce the results of the experiment
    parser.add_argument('--inference_form', type=str, default='long', choices=['long', 'short'], help='self rag includes short form inference and long form inference')
    parser.add_argument("--ignore_cont", action="store_true", help="filter out sentences that include [No support / Contradictory] ") 
    parser.add_argument('--use_citation',  action="store_true", help='add citation for responses')
    # config file
    parser.add_argument('--config',type = str, default = "")
    args = parser.parse_args() # args最好写在 main 里面
    over_write_args_from_file(args, args.config)
    return args

if __name__=='__main__':
    args = get_config()
    set_randomSeed(args)
    # rag = NaiveRag(args) 
    # result = rag.inference("What is Henry Feilden's occupation?",mode = 'interact')
    # rag = SelfRag_Original(args)
    # eval_result = rag.inference(mode = 'evaluation') # TODO SelfRag定义好之后，其实可以多次调用 rag.inference(task = 'factscore) 评测不同的
    rag = SelfRag_Reproduction(args)
    result, generation_track = rag.inference("What is Henry Feilden's occupation?",mode = 'interact')
    print(result)
    print(generation_track)

