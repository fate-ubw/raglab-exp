# raglab
![alt text](https://github.com/fate-ubw/raglab-exp/blob/main/figures/image.png)

# 🔨Install environment
- dev environment：pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
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
- 10-samples test is aimed at validating the environment
- run colbert embedding process enwiki-20230401-10samples.tsv
  1. Change root path for variables: `checkpoint`, `index_dbPath`, `collection` in
[wiki2023-10samples_tsv-2-colbert_embedding.py](https://github.com/fate-ubw/raglab-exp/blob/main/preprocess/colbert-wiki2023-preprocess/wiki2023-db_into_tsv-10samples.py). In file paths, colbert encounters many issues when using relative paths to generate embeddings. Therefore, the current version of raglab uses absolute paths. 
  ~~~bash
    # change root path
  checkpoint = '/your_root_path/raglab-exp/model/colbertv2.0'
  index_dbPath = '/your_root_path/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2023-10samples'
  collection = '/your_root_path/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv'
  ~~~
  2. run
  ~~~bash
  cd raglab-exp
  sh run/wiki2023_preprocess/2-wiki2023-10samples_tsv-2-colbert_embedding.sh
  ~~~
- Embedding precess will take around 15mins in first time.
- The first time colbert processes embeddings, it takes a relatively long time because it needs to recompile the `torch_extensions`. However, calling the processed embeddings does not require a long time. If there are no errors and the retrieved text can be printed, it indicates that the environment is correct.
## Test Raglab with 10-samples embedding
- test selfrag  base on 10-samples embedding
- After processing with colbert embeddings, you can start running the algorithms in raglab. All algorithms integrated in raglab include two modes: `interact` and `evaluation`. The test stage demonstrates in `interact` mode, just for fun 🤗.
- Modify the `index_dbPath` and `text_dbPath` in config file:[selfrag_reproduction-interact-short_form-adaptive_retrieval.yaml](https://github.com/fate-ubw/raglab-exp/blob/main/config/selfrag_reproduction/selfrag_reproduction-interact-short_form-adaptive_retrieval.yaml)
  ~~~bash
  index_dbPath: /your_root_path/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2023-10samples
  text_dbPath: /your_root_path/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv
  ~~~
- run [selfrag](https://arxiv.org/abs/2310.11511) (short form & adaptive retrieval) interact mode test 10-samples embedding
  ~~~bash
  cd raglab-exp
  sh run/rag_inference/3-selfrag_reproduction-interact-short_form-adaptive_retrieval.sh
  ~~~
- Congratulations！！！Now you have already know how to run raglab 🌈
- In raglab, each algorithm has 10 queries built-in in interact mode which are sampled from benchmark
## embedding whole wiki2023
- If the 10-samples test is passed successfully, you can proceed with processing wiki2023.
1. preprocess `.db -> .tsv` (Colbert can only read files in .tsv format.)
    ~~~bash
    cd raglab-exp
    sh run/wiki2023_preprocess/3-wiki2023_db-2-tsv.sh
    ~~~
2. `.tsv -> embedding`
  - remember to change the root  path of `checkpoint`, `index_dbPath` and `collection`
    ~~~bash
      # change root path
        checkpoint = '/your_root_path/raglab-exp/model/colbertv2.0'
        index_dbPath = '/your_root_path/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2023-10samples'
        collection = '/your_root_path/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv'
    ~~~
  - run bash script
    ~~~bash
    cd raglab-exp
    sh run/wiki2023_preprocess/4-wiki2023_tsv-2-colbert_embedding.sh
    ~~~
  - Attention: nranks is the parameter for setting the number of GPUs, currently nranks=8.

# Fine tune llama3 & self rag 
- The base models for raglab baseline and selfrag use llama3-instruction-8b. Since selfrag was further fine-tuned on additional data during the fine-tuning stage, in order to make a fair comparison, the baseline model also needs to be fine-tuned.
## download self rag train data
- we get the train data from [selfrag](https://github.com/AkariAsai/self-rag/tree/main)
- google drive [url](https://drive.google.com/file/d/10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk/view)
- download through gdown
  ~~~bash
  cd raglab-exp/data/train_data/
  gdown --id 10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk
  ~~~
## 10-samples test for fintune
- The 10-samples train dataset has been processed, please directly start the bash script to begin testing.
- Note: The test script only uses one GPU
  - full weight requires 80GB VRam GPU
  ~~~bash
  cd raglab-exp
  sh run/rag_train/script_finetune-llama3-baseline-full_weight-10samples.sh
  ~~~
  - LoRA (Low-Rank Adaptation) requires at least 26GB of VRAM
  ~~~bash
  cd raglab-exp
  sh run/rag_train/script_finetune-llama3-baseline-Lora-10samples.sh
  ~~~
- Congratulations！！！You can now start fine-tuning the baseline and selfrag-8b🤖
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
- finetune baseline ues processed data
  ~~~bash
  sh run/rag_train/script_finetune-llama3-baseline-full_weight.sh
  ~~~
## Merge adapter into base model(only Lora need)
- If you run the the lora finetune scripts, finetune.py only outpits tokenizer and adapter_model. Git clone [llama-factory](https://github.com/hiyouga/LLaMA-Factory) to get final model
- modify path in [merge.sh](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/merge_lora/merge.sh)
  ~~~bash 
  CUDA_VISIBLE_DEVICES=0 python ../../src/export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \
    --template default \
    --finetuning_type lora \
    --export_dir ../../models/llama2-7b-sft \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False
  ~~~
- run merge.sh 
  ~~~bash
  cd /workspace/LLaMA-Factory/examples/merge_lora
  sh merge.sh
  ~~~

# Inference experiments
- Since the experiments conducted by raglab are on a very large scale, the [simple_gpu_scheduler](https://github.com/ExpectationMax/simple_gpu_scheduler) needs to be used to automatically allocate GPUs for different bash scripts in Parallel.
- install simple_gpu_scheduler
  ~~~bash
  pip install simple_gpu_scheduler
  ~~~
- run all experiments in one line 😎
  ~~~bash
  cd raglab-exp
  simple_gpu_scheduler --gpus 0,1,2,3,4,5,6,7 < run_all_inference_experiemnts.txt
  ~~~
