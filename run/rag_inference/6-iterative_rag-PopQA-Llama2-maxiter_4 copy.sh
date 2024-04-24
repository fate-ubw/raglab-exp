export CUDA_VISIBLE_DEVICES=3
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python ./main-iterative_rag.py\
    --config ./config/iterative_rag/iterative_rag-PopQA-Llama2-maxiter_.yaml