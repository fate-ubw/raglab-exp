export CUDA_VISIBLE_DEVICES=4
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python ./main-evaluation.py\
    --config ./config/naive_rag/naive_rag-ASQA-Llama2-without_retrieval.yaml