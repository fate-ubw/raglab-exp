export CUDA_VISIBLE_DEVICES=0,1,2,3
python ./main-evaluation.py\
 --config ./config/naive_rag/naive_rag-PopQA-Llama3-70B-baseline-without_retrieval.yaml
