export CUDA_VISIBLE_DEVICES=5,6
python /home/wyd/RagLab-exp/rag/infer_alg/test_programm.py \
    --num_gpu 2 \
    --llm_path /home/wyd/model/llama-7b-hf \
    --generate_maxlength 50 \
    --use_vllm \
    --retriever_path /home/wyd/model/colbertv2.0 \
    --db_path /home/wyd/ColBERT/experiments/notebook \
    --n_docs 5 \

    