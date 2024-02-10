import os
from datetime import datetime
import jsonlines
import json
from ruamel.yaml import YAML
def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def load_evaldataset(input_file):
    if input_file.endswith(".json"):
        eval_dataset = json.load(open(input_file))
    else:
        eval_dataset = load_jsonlines(input_file) # 这一部分拿到的是一个 list of dict 
    # eval_dataset：type：list of dict
    return eval_dataset

def save_inference_result(inference_result, output_dir, llm_path, eval_datapath):
    print('storing result....')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 文件名称
    model_name = os.path.basename(llm_path) 
    input_filename = os.path.basename(eval_datapath) 
    eval_Dataname = os.path.splitext(input_filename)[0] #这个拿到的是dataset 的 name
    time = datetime.now().strftime('%m%d_%H%M') # time 
    output_name = f'infer_output-{eval_Dataname}-{model_name}-{time}.jsonl' #
    output_file = os.path.join(output_dir, output_name)
    # 存储文件
    
    with open(output_file, 'w') as outfile:
        for result in inference_result:
            json.dump(result, outfile)
            outfile.write('\n')
    print('success!')
    print('start evaluation!')


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