export CUDA_VISIBLE_DEVICES=5
python /home/wyd/RagLab-exp/rag/infer_alg/test_programm.py \
    --n_docs 10 \
    --llm_path /home/wyd/model/llama-7b-hf \
    --generate_maxlength 50 \
    --use_vllm  \
    