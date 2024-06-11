import json
from transformers import AutoTokenizer
from tqdm import tqdm
import pdb
# 加载模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained("/workspace/raglab-exp/model/output_models/self_rag_8B")

count_0_1000 = 0
count_1000_2048 = 0
count_2048_4096 = 0
count_4096_6144 = 0
count_above_6144 = 0
count_total = 0
max_len = 0

with open('/workspace/raglab-exp/data/train_data/full_output_1005.jsonl', 'r') as f:
    data = f.readlines()
    total = len(data)
    pbar = tqdm(total=total, desc="Processing data")
    for line in data:
        sample = json.loads(line)
        instruction = sample['instruction']
        output = sample['output']
        text = instruction + " " + output
        pdb.set_trace()
        input_ids = tokenizer.encode(text, return_tensors='pt')[0]
        token_len = len(input_ids)

        max_len = max(max_len,token_len)
        if token_len <= 1000:
            count_0_1000 += 1
        elif token_len <= 2048:
            count_1000_2048 += 1
        elif token_len <= 4096:
            count_2048_4096 += 1
        elif token_len <=6144:
            count_4096_6144
        else:
            count_above_6144 += 1
        count_total += 1
        pbar.update(1)


print(f"Total samples: {count_total}")
print(f"Samples with length 0-1000: {count_0_1000}")
print(f"Samples with length 1000-2048: {count_1000_2048}")
print(f"Samples with length 2048-4096: {count_2048_4096}")
print(f"Samples with length 4096-6144:{count_4096_6144}")
print(f"Samples with length above 6144:{count_above_6144}")
print(f"max len:{max_len}")

