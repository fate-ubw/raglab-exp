export CUDA_VISIBLE_DEVICES=0
python ./motivation_experiments/selfrag_first_token_distribution.py \
    --seed 633 \
    --task ASQA \
    --eval_datapath ./data/eval_datasets/ASQA/asqa_eval_gtr_top100.json \
    --output_dir ./data/eval_results \
    --llm_mode HF_Model \
    --llm_path ./model/selfrag_llama3_8b-epoch_0_1 \
    --dtype half \
    --temperature 0.0 \
    --top_p 1.0