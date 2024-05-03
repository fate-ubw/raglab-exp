export CUDA_VISIBLE_DEVICES=0

python  ./main-evaluation.py\
    --config ./config/query_rewrite_rag/query_rewrite_rag-PopQA-Llama2.yaml
