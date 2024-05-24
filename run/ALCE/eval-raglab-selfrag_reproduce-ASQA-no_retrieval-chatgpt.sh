# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/selfrag_reproduction-ASQA-selfrag_llama3_8B-epoch_0_1-fc-colbert_api-0521_0746_49/rag_output-selfrag_reproduction|ASQA|selfrag_llama3_8B-epoch_0_1-fc|colbert_api|time=0521_0746_49.jsonl' \
    --mauve \
    --qa
