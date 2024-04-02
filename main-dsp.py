import argparse
import random
import torch
import numpy as np
from raglab.rag.infer_alg.dsp_backup_2024_4_1 import Dsp
from utils import over_write_args_from_file

# def set_randomSeed(args):
#     # random seed
#     if args.use_seed == True:
#         random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         np.random.seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)
#         torch.backends.cudnn.deterministic = True

def get_config():
    parser = argparse.ArgumentParser(description="Configuration for DSP")

    # RAG method
    parser.add_argument('--RAG', type=str, help='RAG name')

    # dataset
    parser.add_argument('--task', type=str, help='Task name')
    parser.add_argument('--eval_datapath', type=str, help='Evaluation data path')

    # language model
    parser.add_argument('--model_mode', type=str, help='Model mode (HFModel or OpenAI)')
    parser.add_argument('--llm_path', type=str, help='Path to the local language model')
    parser.add_argument('--llm_api', type=str, help='API language model name')
    parser.add_argument('--api_key', type=str, help='API key for accessing the model')
    parser.add_argument('--api_base', type=str, help='Base URL for the API')
    parser.add_argument("--dtype", type=str, default= "half", help="all base model inference using half(fp16)")
    parser.add_argument('--top_p', type=float, default=1.0, help='top-p of decoding algorithm')
    parser.add_argument('--use_vllm', action = "store_true", help = 'llm generate max length')
    parser.add_argument('--generation_stop', type=str, default='', help='early_stop is one of the setting of generate() function, early_stop to control the outputs of llm')

    # generate parameters
    parser.add_argument('--generate_maxlength', type=int, help='Maximum length for generation')
    parser.add_argument('--temperature', type=float, help='Temperature for generation')
    parser.add_argument('--output_dir', type=str, help='Output directory')

    # # retrieval model
    # parser.add_argument('--retrieval_url', type=str, help='Retrieval model URL')
    # # retrieval parameters
    # parser.add_argument('--passages_per_hop', type=int, help='Number of passages per hop')

    # retrieval model
    parser.add_argument('--retrieval_name', type = str, default = 'colbert', choices = ['colbert','contriever'],help = 'the name of retrieval model')
    parser.add_argument("--index_dbPath", type = str, help = 'path to index database. Index is index and embedding pairs')
    parser.add_argument('--text_dbPath', type = str, help='path to text database')
    parser.add_argument("--retriever_modelPath", type = str, help = 'path to colbert model')
    # retrieval parameters
    parser.add_argument('--nbits', type = int, default = 2, help = 'encode each dimension with n bits')
    parser.add_argument('--num_gpu', type = int, default = 1, help = 'the number of gpu')
    parser.add_argument('--doc_maxlen', type = int, default = 300, help = 'the doc max len decided by the wikidata format, here we set 300')
    parser.add_argument("--n_docs", type= int, default=10, help="Number of documents to retrieve per questions")
   
    # dsp parameters
    parser.add_argument('--inference_CoT', type=bool, help='Whether to use Chain of Thought for inference')
    parser.add_argument('--signature_retrieval', type=bool, help='Whether to use signature retrieval')
    parser.add_argument('--max_hops', type=int, help='Maximum number of hops')
    parser.add_argument('--eval_threads', type=int, help='Number of evaluation threads')

    # evaluate parameters
    parser.add_argument('--metrics', type=str, help='Evaluation metrics')

    # config file
    parser.add_argument('--config',type = str, default = "")
    args = parser.parse_args()
    over_write_args_from_file(args, args.config)
    return args

if __name__=='__main__':
    args = get_config()
    # set_randomSeed(args)
    rag = Dsp(args)
    # rag.test()
    # 1. 这是用来测试单个的
    # result = rag.inference(mode = 'interact', query="How many storeys are in the castle that David Gregory inherited?")
    # 2. 这是用来测试数据集的，但是目前数据集太大，我进行了截取
    result = rag.inference(mode = 'evaluation')
    print(result)
