export CUDA_VISIBLE_DEVICES=3
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python -i ./main-query_rewrite_rag.py\
    --config ./config/query_rewrite_rag/query_rewrite_rag-interact.yaml