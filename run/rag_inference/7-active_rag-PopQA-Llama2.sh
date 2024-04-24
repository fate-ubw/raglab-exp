export CUDA_VISIBLE_DEVICES=5
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python -i ./main-active_rag.py\
    --config ./config/active_rag/active_rag-PopQA-Llama2.yaml