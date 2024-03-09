export CUDA_VISIBLE_DEVICES=7
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python -i ./main-query_rewrite_rag.py\
    --config ./config/query_rewrite_rag/query_rewrite_rag-PopQA-Llama2.yaml
