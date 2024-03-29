import argparse
import pdb
import pudb
import random
import torch
import numpy as np
from raglab.dataset.utils import TASK_LIST
from utils import over_write_args_from_file
from raglab.rag.utils import get_algorithm, ALGOROTHM_LIST

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
    # common config
    parser.add_argument('--seed', type=int, default = 633, help='random  seed')
    parser.add_argument('--use_seed', action= 'store_true', help='this args will control all random seed of torch, numpy and pyhthon operation')
    parser.add_argument('--num_gpu', type = int, default = 1, help = 'the number of gpu')
    
    # evaluation config
    parser.add_argument('--algorithm_name', type=str, default='naive_rag', choices= ALGOROTHM_LIST, help='name of rag algorithm' )
    parser.add_argument('--task', type=str, default='', choices= TASK_LIST,  help='name of evaluation dataset, different task will select different format and instruction')
    parser.add_argument("--eval_datapath", type = str, help = 'path to eval dataset')
    parser.add_argument('--output_dir', type = str, help = 'the output dir of evaluation')

    # llm config
    parser.add_argument("--llm_path", type = str, help = 'path to llm')
    parser.add_argument('--download_dir', type=str, default=".cache",help="specify vllm model download dir")
    parser.add_argument("--world_size",  type=int, default=1,help="world size to use multiple GPUs. world_size will be used in LLM() function")
    parser.add_argument("--dtype", type=str, default= "half", help="all base model inference using half")
    parser.add_argument('--generate_maxlength', type = int, default = 50, help = 'llm generate max length')
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature of decoding algorithm')
    parser.add_argument('--top_p', type=float, default=1.0, help='top-p of decoding algorithm')
    parser.add_argument('--generation_stop', type=str, default='', help='early_stop is one of the setting of generate() function, early_stop to control the outputs of llm')
    parser.add_argument('--use_vllm', action = "store_true", help = 'llm generate max length')
    
    # retrieval config
    parser.add_argument('--realtime_retrieval', action='store_true', help='self rag can use local passages(only)')
    parser.add_argument('--retrieval_name', type = str, default = 'colbert', choices = ['colbert','contriever'],help = 'the name of retrieval model')
    parser.add_argument("--index_dbPath", type = str, help = 'path to index database. Index is index and embedding pairs')
    parser.add_argument('--text_dbPath', type = str, help='path to text database')
    parser.add_argument("--retriever_modelPath", type = str, help = 'path to colbert model')
    parser.add_argument("--n_docs", type= int, default=10, help="Number of documents to retrieve per questions")
    parser.add_argument('--doc_maxlen', type = int, default = 300, help = 'the doc max len decided by the wikidata format, here we set 300')
    parser.add_argument('--nbits', type = int, default = 2, help = 'encode each dimension with n bits')
    
    # contrieval conefig
    parser.add_argument('--projection_size', type = int, default=768, help = 'size of embedding for contrieval')
    parser.add_argument('--n_subquantizers', type = int, default=0, help="Number of subquantizer used for vector quantization, if 0 flat index is used")
    parser.add_argument('--n_bits', type = int, default = 8, help="Number of bits per subquantizer")
    parser.add_argument("--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed")   
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")# no need thie parameter in colbert
    parser.add_argument("--normalize_text", action="store_true", help="normalize text") # no need thie parameter in colbert
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    # rag common config
    parser.add_argument('--use_citation',  action="store_true", help='add citation for responses')
    # TODO Only selfrag algorithm realize use_citation in this verison of raglab. Next version will update citation feature for all algorithm
    # self rag config
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
    parser.add_argument('--inference_form', type=str, default='long_form', choices=['long_form', 'short_form'], help='self rag includes short form inference and long form inference')
    parser.add_argument("--ignore_cont", action="store_true", help="filter out sentences that include [No support / Contradictory] ") 

    # Iterative rag config
    parser.add_argument('--max_iteration', type=int, default=3, help='max number of iteration in Iterative rag')

    # Active rag config
    parser.add_argument('--max_fianl_answer_length', type=int, default=300, help='max length of final answer')
    parser.add_argument('--filter_prob', type=float, default=0.8, help='filter prob is lower probability threshold in paper(https://arxiv.org/abs/2305.06983)')
    parser.add_argument('--masked_prob', type=float, default=0.4, help='masked prob is low-confidence threshold in paper(https://arxiv.org/abs/2305.06983)')

    # dsp config


    # config file
    parser.add_argument('--config',type = str, default = "")
    args = parser.parse_args()
    over_write_args_from_file(args, args.config)
    return args

if __name__=='__main__':
    args = get_config()
    set_randomSeed(args)
    pdb.set_trace()
    rag = get_algorithm(args)
    evaluation_result = rag.inference(mode = 'evaluation')
    print(evaluation_result)