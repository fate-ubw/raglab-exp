export CUDA_VISIBLE_DEVICES=0
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python -i ./main-interact.py\
    --config ./config/active_rag/active_rag-interact-Llama2.yaml
