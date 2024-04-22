export CUDA_VISIBLE_DEVICES=4
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python -i ./main-interact.py\
    --config ./config/naive_rag/naive_rag-interact.yaml
