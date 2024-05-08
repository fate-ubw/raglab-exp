export CUDA_VISIBLE_DEVICES=5
python  /home/wyd/raglab-exp/ALCE/eval.py --f '.././1-eval_output/ASQA/rag_output-selfrag_reproduction|ASQA|selfrag_llama2_7b|contriever|time=0422_0625.jsonl' \
    --mauve \
    --qa  
# --citations \
