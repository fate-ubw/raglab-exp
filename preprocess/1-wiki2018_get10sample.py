# 打开原始文件
import pdb
from tqdm import tqdm
with open('/home/wyd/raglab-exp/data/retrieval/contriever_passages/psgs_w100-debug.tsv', 'r') as file:
    lines = file.readlines()
pdb.set_trace()
# 取前10行
samples = lines[1:11]
wiki_data = [str(int(sample.split('\t',2)[0])-1)+'\t'+sample.split('\t',2)[1]+'\t'+sample.split('\t',2)[2] for sample in samples]

# 写入新文件
with open('/home/wyd/raglab-exp/preprocess/wiki2018_10samples_idxfrom_0.tsv', 'w') as file:
    file.writelines(wiki_data)