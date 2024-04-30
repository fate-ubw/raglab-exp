# 打开原始文件
import pdb
with open('/home/wyd/raglab-exp/data/retrieval/colbertv2.0_passages/lotte/lifestyle/dev/collection.tsv', 'r') as file:
    lines = file.readlines()

# 取前10行
sample = lines[:10]
# 写入新文件
with open('/home/wyd/raglab-exp/preprocess/sample.tsv', 'w') as file:
    file.writelines(sample)