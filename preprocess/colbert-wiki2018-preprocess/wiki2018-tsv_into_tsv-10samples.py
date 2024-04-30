import pdb
with open('/home/wyd/raglab-exp/data/retrieval/contriever_passages/psgs_w100-debug.tsv', 'r') as file:
    lines = file.readlines()


samples = lines[1:11]
# remove first line
wiki_data = [str(int(sample.split('\t',2)[0])-1)+'\t'+sample.split('\t',2)[1]+'\t'+sample.split('\t',2)[2] for sample in samples]


with open('/home/wyd/raglab-exp/preprocess/wiki2018_10samples_idxfrom_0.tsv', 'w') as file:
    file.writelines(wiki_data)