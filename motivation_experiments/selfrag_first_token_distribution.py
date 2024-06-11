import argparse
import random
import json
from tqdm import tqdm
import numpy as np
import torch
import pdb
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath('/home/wyd/zxw/raglab-exp/motivation_experiments')) # 这个路径是必须的，因为这样才能定位到根目录raglab-exp
sys.path.append(BASE_DIR)
print(sys.path)
from raglab.dataset import get_dataset, TASK_LIST
from raglab.language_model import HF_VLLM, Lora_Model
from raglab.instruction_lab import INSTRUCTION_LAB
from raglab.rag.infer_alg.self_rag_reproduction.utils import load_special_tokens

def set_randomSeed(args):
    # random seed
    if args.use_seed == True:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

def first_token_prob(args):
    llm = steup_llm(args)
    EvalData =  get_dataset(args)
    eval_dataset = EvalData.load_dataset()
    retrieval_tokens, _, _, _ = load_special_tokens(llm.tokenizer, use_grounding=True, use_utility=True)
    json_data = []
    for _, eval_data in enumerate(tqdm(eval_dataset)):
        eval_data = EvalData.preprocess(eval_data) # some dataset need preprocess such as: arc_challenge
        target_instruction = find_instruction('selfrag_reproduction-read', args.task)
        input = target_instruction.format_map({'query': eval_data[EvalData.InputStruction.question]})
        outputs_list = llm.generate([input])
        Outputs = outputs_list[0]
        generated_text = Outputs.text
        pred_log_probs = Outputs.logprobs
        score_dict = {}
        for tok, id in retrieval_tokens.items():
            prob = pred_log_probs[0][id] 
            score_dict[tok] = np.exp(prob)
        retrieve_prob = score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])  
        json_data.append({
                    "prompt": input,
                    "generated_text": generated_text,
                    "first_tokens_logits": score_dict,
                    "retrieval_ratio": retrieve_prob
                })
    # -> end of for loop
    llm_name = os.path.basename(args.llm_path.rstrip('/'))
    store_json(json_data, f"./{args.output_dir}/first_token_distribution-{llm_name}-{args.task}.json")
    # statis_distribution
    statistic_results = firts_token_statistic(json_data)
    store_json(statistic_results, f"./{args.output_dir}/first_token_distribution-{llm_name}-{args.task}.json.statistic_results")
    print(f'success! files save path: ./{args.output_dir}/first_token_distribution-{llm_name}-{args.task}.json.statistic_results')
    return statistic_results

def firts_token_statistic(input_data):
    statistic_first_token = {'[No Retrieval]': 0, '[Retrieval]': 0, 'others': 0} 
    statistic_Magnitude = {'1e-5':0,'1e-4':0,'1e-3':0,'1e-2':0,'1e-1':0,'1':0}
    for item in input_data: 
        generated_text = item['generated_text'] 
        no_retrieval_token = generated_text[0:14]
        retrieval_token = generated_text[0:11] 

        if retrieval_token == '[Retrieval]':
            statistic_first_token['[Retrieval]'] += 1
        elif no_retrieval_token == '[No Retrieval]':
            statistic_first_token['[No Retrieval]'] += 1
        else:
            statistic_first_token['others'] += 1
        retrieval_tokens_probs = item['first_tokens_logits'] 
        if retrieval_tokens_probs['[No Retrieval]'] <= 1e-5 and retrieval_tokens_probs['[Retrieval]'] <= 1e-5:
            statistic_Magnitude['1e-5'] +=1
        elif retrieval_tokens_probs['[No Retrieval]'] <= 1e-4 and retrieval_tokens_probs['[Retrieval]'] <= 1e-4:
            statistic_Magnitude['1e-4'] += 1
        elif retrieval_tokens_probs['[No Retrieval]'] <= 1e-3 and retrieval_tokens_probs['[Retrieval]'] <= 1e-3:
            statistic_Magnitude['1e-3'] += 1
        elif retrieval_tokens_probs['[No Retrieval]'] <= 1e-2 and retrieval_tokens_probs['[Retrieval]'] <= 1e-2:
            statistic_Magnitude['1e-2'] += 1           
        elif retrieval_tokens_probs['[No Retrieval]'] <= 1e-1 and retrieval_tokens_probs['[Retrieval]'] <= 1e-1:
            statistic_Magnitude['1e-1'] += 1           
        else:
            statistic_Magnitude['1'] += 1  
    return statistic_first_token, statistic_Magnitude

def steup_llm(args):
    if args.llm_mode == 'HF_Model':
        llm = HF_VLLM(args)
        llm.load_model() # load_model() will load local model and tokenizer  
    elif args.llm_mode == "Lora_Model":
        llm = Lora_Model(args)
        llm.load_model() #  load_model() will load base model and lora adapter then merged by peft to get complete model
    else:
        raise LanguageModelError("Language model must be huggingface or openai api.")
    return llm

def store_json(json_data, json_path):
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json_file.seek(0)
        json_file.truncate()
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)
    json_file.close()

def find_instruction( rag_name:str, dataset_name:str) -> str:
    for instruction in INSTRUCTION_LAB:
        if instruction['rag_name'] == rag_name and instruction['dataset_name'] == dataset_name:
            target_instruction = instruction['instruction']
            break
    if target_instruction == '':
        raise InstructionNotFoundError('Instruction name not recognized. Please provide a valid instruction name.')
    return target_instruction

class InstructionNotFoundError(Exception):
    pass

class LanguageModelError(Exception):
    pass

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default = 633, help='random  seed')
    parser.add_argument('--use_seed', type=int, default=True, help='this args will control all random seed of torch, numpy and pyhthon operation')
    parser.add_argument('--task', type=str, default='', choices= TASK_LIST,  help='name of evaluation dataset, different task will select different format and instruction')
    parser.add_argument("--eval_datapath", type = str, help = 'path to eval dataset')
    parser.add_argument('--output_dir', type = str, default='./',help = 'the output dir of evaluation')
    parser.add_argument("--llm_path", type = str, help = 'path to llm or lora adapter')
    parser.add_argument('--llm_mode', type = str, default='HF_Model', choices=['HF_Model', 'Openai_api', 'Lora_Model'], help='flag of language or api')
    parser.add_argument('--basemodel_path', type = str, help = 'path of lora base model, only Lora need base model')
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature of decoding algorithm')
    parser.add_argument('--top_p', type=float, default=1.0, help='top-p of decoding algorithm')
    parser.add_argument('--generate_maxlength', type = int, default = 50, help = 'llm generate max length')
    parser.add_argument("--dtype", type=str, default= "half", help="all base model inference using half")
    parser.add_argument('--generation_stop', type=str, default='', help='early_stop is one of the setting of generate() function, early_stop to control the outputs of llm')
    parser.add_argument('--include_stop_token', type=int, default=False, help='"include_stop_token" controls whether the generated text output should include the provided stop string.')
    parser.add_argument('--use_chat_template', type = int, default=False, help = 'llama2-chat and llama3-instruction ues official chat template will get a better performance, but finetune model will mess up by this template')
    args = parser.parse_args()
    set_randomSeed(args)
    statistic_results =  first_token_prob(args)
    print(statistic_results)