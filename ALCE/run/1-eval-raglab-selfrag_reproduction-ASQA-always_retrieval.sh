export CUDA_VISIBLE_DEVICES=6
python -i /home/wyd/RagLab/ALCE/eval.py --f .././1-eval_output/infer_output-asqa_eval_gtr_top100-selfrag_llama2_7b-0229_1006-ALCE.jsonl \
   --citations \
    --mauve \
    --qa