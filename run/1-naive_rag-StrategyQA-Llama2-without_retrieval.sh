export CUDA_VISIBLE_DEVICES=5
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python ./main-evaluation.py\
    --config ./config/naive_rag/naive_rag-StrategyQA-Llama2-without_retrieval.yaml