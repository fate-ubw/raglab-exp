export CUDA_VISIBLE_DEVICES=3
python -i /home/wyd/RagLab/ALCE/eval.py --f ../.././1-eval_output/infer_output-asqa_eval_gtr_top100--0222_0234.jsonl \
   --citations \
    --mauve \
    --qa