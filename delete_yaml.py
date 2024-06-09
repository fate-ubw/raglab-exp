import os

def delete_llama3_instruct_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'Llama3-Instruct' in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# 指定要处理的目录路径
directory = './run/rag_inference'

# 删除名字中包含 "Llama3-Instruct" 的文件
delete_llama3_instruct_files(directory)