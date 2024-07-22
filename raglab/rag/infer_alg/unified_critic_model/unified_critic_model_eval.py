import os
import sys
import re
import time
import json
import jsonlines
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath('/home/wyd/zxw/raglab-exp/raglab'))
sys.path.append(BASE_DIR)

from raglab.language_model import OpenaiModel, HF_Model, HF_VLLM, Lora_Model
import pdb


PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def load_dataset(self)-> list[dict]:
    if self.eval_datapath.endswith(".json"):
        eval_dataset = json.load(open(self.eval_datapath))
    else:
        eval_dataset = load_jsonlines(self.eval_datapath)
    return eval_dataset

def load_jsonlines(file:str)-> list[dict]:
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst 

class CriticModelEval:

    def __init__(self, args):
        self.eval_datapath = args.eval_datapath
        self.output_dir = args.output_dir
        # llm config
        self.llm_mode = args.llm_mode
        self.llm_path = args.llm_path
        self.dtype = args.dtype
        self.temperature = args.temperature
        self.generate_maxlength = args.generate_maxlength
        self.top_p = args.top_p
        self.generation_stop = args.generation_stop
        self.include_stop_token = args.include_stop_token
        self.use_chat_template = args.use_chat_template
        self.use_vllm = args.use_vllm
        # setup model and database 
        self.llm = self.steup_llm(args)
        self.start_time = None

    def eval(self) -> tuple[str, dict]:
        self.start_time = time.time()
        evaluation_result = defaultdict(int)
        task_counts = defaultdict(int)
        test_data = load_dataset(args)
        # Set up progress bar
        total_items = len(test_data)
        pbar = tqdm(total=total_items, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        # Set up logging
        log_interval = min(1, total_items // 100)  # Update log every 1% of progress
        last_log_time = time.time()
        for idx, item in enumerate(test_data):
            task = item['task']
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            input = prompt_input.format_map(item) if item.get("input", "") != "" else prompt_no_input.format_map(item)
            outputlist = self.llm.generate(input)
            output = outputlist[0].text
            item['generation'] = output
            label = item['output']
            result = self.evaluator(output, label, task)
            if task not in evaluation_result:
                evaluation_result[task] = 0
                task_counts[task] = 0

            evaluation_result[task] += result
            task_counts[task] += 1
            pbar.update(1)
            #  Log accuracy every log_interval items or if it's the last item
            if (idx + 1) % log_interval == 0 or idx == total_items - 1:
                self.log_accuracy(evaluation_result, task_counts, idx + 1, total_items, last_log_time)
                last_log_time = time.time()

        pbar.close()
        # Calculate accuracy for each task
        accuracy_result = {task: evaluation_result[task] / task_counts[task] for task in evaluation_result}
        self.save_results(test_data)
        return accuracy_result

    def log_accuracy(self, evaluation_result, task_counts, current_item, total_items, last_log_time):
        os.system('clear' if os.name == 'posix' else 'cls')
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        estimated_total_time = (elapsed_time / current_item) * total_items if current_item > 0 else 0

        print(f"\n{'=' * 50}")
        print(f"Progress: {current_item}/{total_items} ({current_item/total_items:.2%})")
        print(f"Time: {self.format_time(elapsed_time)} / {self.format_time(estimated_total_time)}")

        print(f"{'=' * 50}")
        print(f"{'Task':<20}{'Accuracy':<10}{'Count':<10}")
        print(f"{'-' * 50}")
        for task in evaluation_result:
            accuracy = evaluation_result[task] / task_counts[task] if task_counts[task] > 0 else 0
            print(f"{task:<20}{accuracy:>.2%}{task_counts[task]:>10}")
        print(f"{'=' * 50}")
        print(f"Time since last update: {time.time() - last_log_time:.2f} seconds")
        print(f"{'=' * 50}\n")

    @staticmethod
    def format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def steup_llm(self, args):
        if self.llm_mode == 'HF_Model':
            if self.use_vllm:
                llm = HF_VLLM(args)
                llm.load_model() # load_model() will load local model and tokenizer  
            else:
                llm = HF_Model(args)
                llm.load_model() # load_model() will load local model and tokenizer
        elif self.llm_mode == "Lora_Model":
            llm = Lora_Model(args)
            llm.load_model() #  load_model() will load base model and lora adapter then merged by peft to get complete model
        elif self.llm_mode == 'Openai_api':
            llm = OpenaiModel(args)
            llm.load_model() # load_model() will load api configs and tiktoken
        else:
            raise LanguageModelError("Language model must be huggingface or openai api.")
        return llm

    def evaluator(self, output, label, task):
        result = None
        task_list = ['retrieval', 'relevance', 'groudness', 'utility', 'improvement_answer', 'pair_wise']

        def extract_keyword_with_brackets(text, keywords):
            for keyword in keywords:
                pattern = rf'\[{re.escape(keyword)}\]'
                match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
                if match:
                    return match.group()
            return None

        def extract_keyword_pair_wise(text, keywords):
            for keyword in keywords:
                pattern = rf'{re.escape(keyword)}'
                match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
                if match:
                    return match.group()
            return None

        if task in ['retrieval', 'relevance', 'groudness', 'utility']:
            keywords = {
                'retrieval': ['Retrieval', 'No Retrieval'],
                'relevance': ['Relevant', 'Irrelevant'],
                'groudness': ['No support / Contradictory', 'Partially supported', 'Fully supported'],
                'utility': ['Utility:1', 'Utility:2']
            }[task]
            label_keyword = extract_keyword_with_brackets(label, keywords)
            output_keyword = extract_keyword_with_brackets(output, keywords)
            if output_keyword is None:
                result = 0
            else:
                result = 1 if output_keyword == label_keyword else 0
        elif task == 'improvement_answer':
            result = 1 if label.lower() in output.lower() else 0

        elif task == 'pair_wise':
            keywords = ['### Evaluation: 1', '### Evaluation: 2', '### Evaluation: tie']
            label_keyword = extract_keyword_pair_wise(label, keywords)
            output_keyword = extract_keyword_pair_wise(output, keywords)
            pdb.set_trace()
            if label_keyword is None or output_keyword is None:
                result = 0
            else:
                label_value = label_keyword.split()[-1]
                output_value = output_keyword.split()[-1]
                result = 1 if label_value == output_value else 0
        else:
            assert False, f"Unknown task: {task}"
        return result

    def save_results(self, data):
        with jsonlines.open(self.output_dir, mode='w') as writer:
            writer.write_all(data)


class LanguageModelError(Exception):
    pass


if __name__ == '__main__':

    class Args():
        def __init__(self):
            self.eval_datapath = './data/collected_data/test_20w.jsonl'
            self.output_dir = './data/collected_data/test_20w_eval_result.jsonl'
            # llm config
            self.llm_path = './model/output_models/unified-Critic-8B-baseline_2w'    
            self.llm_mode = 'HF_Model'
            self.dtype = 'half'
            self.temperature = 0.0
            self.top_p = 1.0
            self.generate_maxlength =300
            self.generation_stop = ''
            self.include_stop_token = False
            self.use_chat_template = False
            self.use_vllm = True

    args = Args()
    unified_critc_model = CriticModelEval(args)
    evaluation_result = unified_critc_model.eval()
    pprint(evaluation_result)