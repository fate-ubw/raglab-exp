import os
from datetime import datetime
import jsonlines
import json

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

