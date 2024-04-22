export CUDA_VISIBLE_DEVICES=5
python -i /home/wyd/raglab-exp/ALCE/eval.py --f '.././1-eval_output/infer_output-asqa_eval_gtr_top100--0221_2350.jsonl' \
    --mauve \
    --qa  
# --citations \
