# raglab
![alt text](https://github.com/fate-ubw/raglab-exp/blob/main/figures/image.png)

# 🔨Install environment
- 机器配置：pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
- [install miniconda](https://docs.anaconda.com/free/miniconda/index.html)

- git clone raglab
~~~bash
git clone https://github.com/fate-ubw/raglab-exp.git
~~~
- create environment from yml 
~~~bash
cd raglab-exp
conda create -f environment.yml
~~~
- install flash-attn, en_core_web_sm, punkt manually
~~~bash
pip install flash-attn==2.2
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
~~~

# 🤖Model
- raglab need llama2-7b, llama3-instruction, colbertv2.0, selfrag
~~~bash
cd raglab-exp
mkdir model
cd model
mkdir output_models
mkdir Llama-2-7b-chat-hf
mkdir Meta-Llama-3-8B-Instruct
mkdir selfrag_llama2_7b
mkdir colbertv2.0
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir Meta-Llama-3-8B-Instruct/
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir Llama-2-7b-chat-hf/
huggingface-cli download selfrag/selfrag_llama2_7b --local-dir selfrag_llama2_7b
huggingface-cli download colbert-ir/colbertv2.0 --local-dir colbertv2.0/
~~~

# 💽process wiki2023 as vector database

## download wiki2023 raw data
- current version of raglab use wiki2023 as database
- we get source wiki2023 get from [factscore](https://github.com/shmsw25/FActScore)
  - method1: url for download wiki2023:[google_drive](https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view) 
  - method2: install throuth gdown 
  ~~~bash
  cd raglab-exp/data/retrieval/colbertv2.0_passages
  mkdir wiki2023
  pip install gdown
  gdown --id 1mekls6OGOKLmt7gYtHs0WGf5oTamTNat
  ~~~
## 10-samples test
- use 10-samples test environment
- run colbert embedding process enwiki-20230401-10samples.tsv
- 第一步：首先需要修改raglab-exp/preprocess/colbert-wiki2023-preprocess/wiki2023-10samples_tsv-2-colbert_embedding.py 文件中的路径，colbert 在生成 embedding 的时候使用相对路径会造成很多问题，固当前版本 raglab 使用绝对路径
~~~bash
  # change root path
    checkpoint = '/your_root_path/raglab-exp/model/colbertv2.0'
    index_dbPath = '/your_root_path/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2023-10samples'
    collection = '/your_root_path/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv'
~~~
- 第二步：启动脚本开始处理
~~~bash
cd raglab-exp
sh run/wiki2023_preprocess/2-wiki2023-10samples_tsv-2-colbert_embedding.sh
~~~
- this will take around 15mins. 
- 10-samples test 目标是为了验证环境，colbert embedding第一次处理花费时间较大，因为需要重新编译 torch_extensions。调用处理好的 embedding 则无需花费很长时间
-  若无报错并且可以打检索到的文本，则说明环境正确
## Test Raglab with 10-samples embedding
- here we test naive rag base  10-samples embedding
- 经过 colbert embedding 的处理之后就可以开始运行 raglab 中的算法了，raglab集成的所有算法都包含 `interact` 和 `evaluation` 两个 mode，test 阶段展示 `interact mode`
- 修改`/workspace/raglab-exp/config/selfrag_reproduction/selfrag_reproduction-interact-short_form-adaptive_retrieval.yaml`文件中 `index_dbPath` 和 `text_dbPath`. 注意 colbert 必须使用绝对路径
~~~bash
# 其他参数不需要修改
index_dbPath: /your_root_path/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2023-10samples
text_dbPath: /your_root_path/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv
~~~
- run selfrag (short form & adaptive retrieval) interact mode test 10-samples embedding
- you can also run other algorithms in interaction mode
~~~bash
cd raglab-exp
sh run/rag_inference/3-selfrag_reproduction-interact-short_form-adaptive_retrieval.sh
~~~
- In raglab, all algorithms in interact mode return two variables: response and generation_track. Moreover, each algorithm has 10 queries built-in in interact mode which are sampled from benchmark
## embedding whole wiki2023
- 如果顺利通过 10-samples  test 则可以进行 wiki2023 的处理
- preprocess `.db -> .tsv` (colbert 只能读取.tsv格式的文件)
~~~bash
cd raglab-exp
sh run/wiki2023_preprocess/3-wiki2023_db-2-tsv.sh
~~~
- `.tsv -> embedding`
~~~bash
cd raglab-exp
sh run/wiki2023_preprocess/2-wiki2023-10samples_tsv-2-colbert_embedding.sh
~~~
- attention：nranks 为设定 gpu 数量参数，当前nranks=8

# Fine tune llama3 & self rag 
- raglab baseline 和 selfrag 的基座模型采用 `llama3-instruction-8b`。由于 self rag 在微调阶段多训练了一部分数据，为了公平比较 baseline model 同样需要进行微调 
## download self rag train data
- we get the train data from [selfrag](https://github.com/AkariAsai/self-rag/tree/main)
- google drive [url](https://drive.google.com/file/d/10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk/view)
- download through gdown
~~~bash
cd raglab-exp/data/train_data/
gdown --id 10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk
~~~
## 10-samples test for fintune
- 10samples train dataset 已经准备好了，请直接启动 bash 脚本开始测试
- 注意：测试脚本仅启动一张 gpu 进行全参数finetune 
~~~bash
cd raglab-exp
sh run/rag_train/script_finetune-llama3-baseline-full_weight-10samples.sh
~~~
- 若 10-samples test顺利通过，则证明环境无误
## finetune self rag 8b
- finetune directly
~~~bash
cd raglab-exp
sh run/rag_train/script_finetune-selfrag_8b-full_weight.sh
~~~
## finetune llama3-7b-instruction as baseline
- preprocess train data. Train data for baseline model need remove special tokens.
~~~bash
cd raglab-exp
sh run/traindataset_preprocess/selfrag_traindata-remove_special_tokens.sh
~~~
- then you will get baseline train_data without special token (what is specal token? Anawer: special tokens is a concept proposed by SelfRAG)
- finttune baseline ues processed data
~~~bash
 sh run/rag_train/script_finetune-llama3-baseline-full_weight.sh
~~~
# Inference exp
- 完成上述实验之后即可开始所有算法的推理
- baseline 的 temperature 以及 top_p 参数还需要讨论
- 因为需要推理的脚本非常多，需要使用 gpu 调度算法[simple_gpu_scheduler](https://github.com/ExpectationMax/simple_gpu_scheduler)来实现一个脚本调用所有的 bash 脚本进行所有的实验