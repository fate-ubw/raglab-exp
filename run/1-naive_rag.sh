export CUDA_VISIBLE_DEVICES=2
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python ./main-NaiveRag.py\
    --config ./config/naive_rag/naive_rag-interact.yaml