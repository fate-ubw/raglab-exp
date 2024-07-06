import os
import json
import argparse
from typing import List, Dict, Any
import random
from raglab.data_collector import DatasetCollector, CriticModelDatasetCollector

import pdb

class InvalidCollector(Exception):
    pass

def collect_datasets(args, dataset_config):
    all_train_data = []
    all_test_data = []

    for dataset_name, config in dataset_config.items():
        print(f"Collecting data for {dataset_name}")
        args.dataset_name  = dataset_name

        if args.colletor_method:
            collector = CriticModelDatasetCollector(args)
        else:
            collector = DatasetCollector(args)
        
        train_data, test_data = collector.collect_data(
            split=config['split'],
            n=config['samples'],
            format=config['format'],
            test_ratio=config['test_ratio']
        )
        
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

        print(f"Collected {len(train_data)} training samples and {len(test_data)} test samples for {dataset_name}")

    print(f"Total collected: {len(all_train_data)} training samples and {len(all_test_data)} test samples")

    return all_train_data, all_test_data

def save_jsonl(data: List[Dict[str, Any]], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect and process dataset samples.")
    # collecter config
    parser.add_argument('--colletor_method', type=str,choices=['base', 'selfrag-critic'], default='base', help='diff method of collect  trian data for sft')
    parser.add_argument("--base_path", default="/home/wyd/FlashRAG/dataset", help="Base path for dataset directories")
    parser.add_argument("--output", default="collected_data.jsonl", help="Output file name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # language model
    parser.add_argument('--llm_mode', type = str, default='HF_Model', choices=['HF_Model','Openai_api', 'Lora_Model'], help='flag of language or api')
    parser.add_argument("--llm_path", type = str, help = 'path to llm')
    parser.add_argument("--dtype", type=str, default= "half", help="all base model inference using half(fp16)")
    parser.add_argument('--generate_maxlength', type = int, default = 50, help = 'llm generate max length')
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature of decoding algorithm')
    parser.add_argument('--top_p', type=float, default=1.0, help='top-p of decoding algorithm')
    parser.add_argument('--generation_stop', type=str, default='', help='early_stop is one of the setting of generate() function, early_stop to control the outputs of llm')

    # api config
    parser.add_argument('--llm_name', type=str, default='gpt-3.5-turbo', help='language model name of openai api')
    parser.add_argument('--llm_api', type=str, help='API language model name')
    parser.add_argument('--api_key', type=str, help='API key for accessing the model')
    parser.add_argument('--api_base', type=str, help='Base URL for the API')
    parser.add_argument('--api_key_path', type=str, help='path of .txt which save api_key for openai api')
    parser.add_argument('--api_logprobs', type=int, default = False, help='Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.')
    parser.add_argument('--api_top_logprobs', type=int, default=1, help='An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.')
    # retrieval config
    parser.add_argument('--retrieval_name', type = str, default = 'colbert_api', choices = ['colbert_api'], help = 'name of retrieval model')
    parser.add_argument("--n_docs", type= int, default=10, help="Number of documents to retrieve per questions")
    return parser.parse_args()

def main():
    args = parse_arguments()
    random.seed(args.seed)
    dataset_config = {
        # flashrag format dataset
        # "2wikimultihopqa": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "ambig_qa": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "arc": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "asqa": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "ay2": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "bamboogle": {"samples": 1, "split": "test", "format": "flashrag", "test_ratio": 0.1},
        # "boolq": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "commense_qa": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "curatedtrec": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "eli5": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "fever": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "fermi": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "hellaswag": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "hotpotqa": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "mmlu": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "ms_marco": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # "musique": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "narrative_qa": {"samples": 1, "split": "test", "format": "flashrag", "test_ratio": 0.1},
        "nq": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "openbookqa": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "piqa": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "popqa": {"samples": 1, "split": "test", "format": "flashrag", "test_ratio": 0.1},
        "quartz": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "retrieval-corpus": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "sharegpt": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "siqa": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "squad": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "trex": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "triviaqa": {"samples": 1, "split": "dev", "format": "flashrag", "test_ratio": 0.1},
        "truthful_qa": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "web_questions": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "wiki_qa": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "wikiasp": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "wned": {"samples": 1, "split": "dev", "format": "flashrag", "test_ratio": 0.1},
        "wow": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "zsre": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        # alpaca format
        "baize":{"samples": 1, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.1},
        "code_alpaca": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "cot": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "dolly": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "flan_v2": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "gpt4_alpaca": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "hard_coded": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "lima": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "oasst1": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "open_orca": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "science": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "self_instruct": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "stanford_alpaca": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "tulu_v1": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "tulu_v2": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "unnatural_instructions": {"samples": 1, "split": "train", "format": "flashrag", "test_ratio": 0.1},
        "wizardlm":{"samples": 1, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.1},
    }

    train_data, test_data = collect_datasets(args, dataset_config)
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    # Save train data
    train_output_path = os.path.join(args.output, "train.jsonl")
    save_jsonl(train_data, train_output_path)
    print(f"Saved {len(train_data)} training samples to {train_output_path}")
    # Save test data
    test_output_path = os.path.join(args.output, "test.jsonl")
    save_jsonl(test_data, test_output_path)
    print(f"Saved {len(test_data)} test samples to {test_output_path}")
    # 
    print("Data collection and storage complete.")

if __name__ == "__main__":
    main()