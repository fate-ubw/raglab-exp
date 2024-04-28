export CUDA_VISIBLE_DEVICES=0
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python ./main-query_rewrite_rag.py\
    --config ./config/query_rewrite_rag/query_rewrite_rag-PopQA-PandaLM.yaml
