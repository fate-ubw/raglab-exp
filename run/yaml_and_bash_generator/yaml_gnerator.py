import os
import re
import shutil

def copy_and_modify_files(folder_path):
   # Traverse all files in the folder
   for root, dirs, files in os.walk(folder_path):
       for file_name in files:
           # Check if the file ends with .yaml and the file name contains Llama3-baseline
           if file_name.endswith('.yaml') and 'Llama3-baseline' in file_name:
               # Get the full path of the file
               file_path = os.path.join(root, file_name)
               # Construct the new file name
               new_file_name = re.sub(r'Llama3-baseline', r'Llama3-70B-baseline', file_name)
               # Copy the file and rename it
               new_file_path = os.path.join(root, new_file_name)
               shutil.copy(file_path, new_file_path)
               # Modify the file content
               with open(new_file_path, 'r') as file:
                   file_content = file.read()
               
               modified_content = re.sub(r'llm_mode: HF_Model\nllm_path: ./model/selfrag_llama3_8b-epoch_0_1', r'llm_mode: Lora_Model\nllm_path: ./model/selfrag_llama3_70B-adapter\nbasemodel_path: ./model/Meta-Llama-3-70B', file_content)

               with open(new_file_path, 'w') as file:
                   file.write(modified_content)

# Example usage
copy_and_modify_files('./config/selfrag_reproduction')