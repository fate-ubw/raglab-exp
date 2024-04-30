# export CUDA_VISIBLE_DEVICES=0
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python  ./main-interact.py\
    --config ./config/naive_rag/naive_rag-interact-without_retrieval.yaml
