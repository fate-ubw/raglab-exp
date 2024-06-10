export CUDA_VISIBLE_DEVICES=1,2
python ./motivation_experiments/selfrag_first_token_distribution.py \
    --seed 633 \
    --task ASQA \
    --eval_datapath ./data/eval_datasets/ASQA/asqa_eval_gtr_top100.json \
    --output_dir ./data/eval_results \
    --llm_mode Lora_Model \
    --llm_path ./model/selfrag_llama3_70B-adapter \
    --dtype half \
    --basemodel_path ./model/Meta-Llama-3-70B \
    --temperature 0.0 \
    --top_p 1.0