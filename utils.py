from ruamel.yaml import YAML
from raglab.rag.infer_alg.self_rag_original import SelfRag_Original
from raglab.rag.infer_alg.self_rag_reproduction import SelfRag_Reproduction
def get_algorithm(args):
    pass
    

def over_write_args_from_file(args, yml): # 后面这个 yaml 文件是直接从 args load 进来的。
    """
    overwrite arguments according to config file
    """
    # args = Namespace(num_gpu=1, output_dir=None, mode='interact', llm_path=None, db_path=None, eval_datapath=None, retriever_path=None, generate_maxlength=50, n_docs=10, use_vllm=False, doc_maxlen=300, nbits=2, c='/home/wyd/RagLab-exp/rag/infer_alg/naive_rag.yaml')
    # args 非常像一个 dict 类，
    if yml == '':
        return
    yaml = YAML(typ='rt') # rt is (round-trip) mode
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read())
        # {'num_gpu': 2, 'output_dir': '/home/wyd/RagLab-exp/1-eval_output', 'mode': 'interact', 'llm_path': '/home/wyd/model/llama-7b-hf', 'db_path': '/home/wyd/ColBERT/experiments/notebook', 'eval_datapath': '/home/wyd/data/1-self_rag/1-eval_data/factscore_unlabeled_alpaca_13b_retrieval_10samples.jsonl', 'retriever_path': '/home/wyd/model/colbertv2.0', 'generate_maxlength': 500, 'n_docs': 5, 'use_vllm': None}
        for k in dic:
            setattr(args, k, dic[k])