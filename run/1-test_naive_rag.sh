export CUDA_VISIBLE_DEVICES=5,6
python /home/wyd/RagLab-exp/rag/infer_alg/test_programm.py \
    --num_gpu 2 \
    --output_dir /home/wyd/RagLab-exp/1-eval_output\
    --mode interact \
    --llm_path /home/wyd/model/llama-7b-hf \
    --db_path /home/wyd/ColBERT/experiments/notebook \
    --eval_datapath /home/wyd/data/1-self_rag/1-eval_data/popqa_longtail_w_gs.jsonl \
    --retriever_path /home/wyd/model/colbertv2.0 \
    --generate_maxlength 500 \
    --n_docs 5 \
    --use_vllm \

    