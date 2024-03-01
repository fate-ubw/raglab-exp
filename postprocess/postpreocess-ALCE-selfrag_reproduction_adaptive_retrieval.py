import json
from tqdm import tqdm
# Read the JSONL file
import pudb
input_file_path = '/home/wyd/raglab-exp/1-eval_output/infer_output-asqa_eval_gtr_top100-selfrag_llama2_7b-0229_1114.jsonl'  # Replace with your file path
output_file_path = '/home/wyd/raglab-exp/1-eval_output/infer_output-asqa_eval_gtr_top100-selfrag_llama2_7b-0229_1114-ALCE.jsonl'  # Replace with your desired output file path

def replace_key_in_docs_and_save(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        # Read the JSON Lines file
        data = [json.loads(line) for line in file]

    # Extract data from the "data" key
    data_entries = data[0].get("data", [])

    # Iterate through each data entry
    for entry in data_entries:
        if "docs" in entry and isinstance(entry["docs"], list):
            # Iterate through each dictionary in the "docs" list
            for doc in entry["docs"]:
                if "content" in doc:
                    # Replace "content" with "text"
                    doc["text"] = doc.pop("content")

    # Write the updated data to a new JSON Lines file
    with open(output_file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

# Replace key in the "docs" list and save to a new JSON Lines file
replace_key_in_docs_and_save(input_file_path, output_file_path)
