import json
import re
from tqdm import tqdm
import pdb

def process(pred) -> str:
    # remove all special tokens
    special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", 
                      "[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]",
                      "[Irrelevant]", "[Relevant]",
                      "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    for item in special_tokens:
        if item in pred:
            pred = pred.replace(item, "")
    return pred

# read JSONL
with open('./data/train_data/full_output_1005.jsonl', 'r') as file:
    data = file.readlines()

processed_data = []
for line in tqdm(data):
    json_data = json.loads(line.strip())
    output = json_data['output']
    # Call the process function to remove special tokens from output
    output_cleaned = process(output)
    json_data['output'] = output_cleaned.strip()
    processed_data.append(json_data)

# Save processed data
with open('./data/train_data/full_output_1005-remove_special_tokens.jsonl', 'w') as file:
    for item in processed_data:
        json.dump(item, file)
        file.write('\n')

print('Data preprocessing completed.')