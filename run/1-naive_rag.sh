export CUDA_VISIBLE_DEVICES=6
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python -i ./main-NaiveRag.py\
    --config ./config/naive_rag-colbert.yaml