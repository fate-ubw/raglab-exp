export CUDA_VISIBLE_DEVICES=1
# export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True
python -i ./main-evaluation.py\
    --config ./config/self_ask/self_ask-PopQA-Llama2.yaml