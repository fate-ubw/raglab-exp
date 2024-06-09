import os
import re
import shutil

def copy_and_modify_files(folder_path):
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # 判断文件是否以.yaml结尾且文件名包含Llama3-baseline
            if file_name.endswith('.yaml') and 'Llama3-baseline' in file_name:
                # 获取文件的完整路径
                file_path = os.path.join(root, file_name)
                # 构建新文件名
                new_file_name = re.sub(r'Llama3-baseline', r'Llama3-Instruct', file_name)
                # 复制文件并重命名
                new_file_path = os.path.join(root, new_file_name)
                shutil.copy(file_path, new_file_path)
                # 修改文件内容
                with open(new_file_path, 'r') as file:
                    file_content = file.read()
                
                modified_content = re.sub(r'llm_path: \./model/Llama3-8B-baseline', r'llm_path: ./model/Llama3-8B-Instruct-baseline', file_content)

                with open(new_file_path, 'w') as file:
                    file.write(modified_content)

# 调用示例
copy_and_modify_files('./config/')
