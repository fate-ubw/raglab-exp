export CUDA_VISIBLE_DEVICES=6
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python ./main-Naive_rag.py\
    --config ./config/naive_rag/naive_rag-PopQA-Llama2.yaml
