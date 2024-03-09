export CUDA_VISIBLE_DEVICES=6
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python -i ./main-query_rewrite_rag.py\
    --config ./config/query_rewrite_rag/query_rewrite_rag-PopQA-PandaLM-without_retrieval.yaml
