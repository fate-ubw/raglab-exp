export CUDA_VISIBLE_DEVICES=0
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python -i ./main-evaluation.py\
    --config ./config/query_rewrite_rag/query_rewrite_rag-PopQA-Llama2.yaml
