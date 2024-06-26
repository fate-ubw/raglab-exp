export CUDA_VISIBLE_DEVICES=1
python  ./ALCE/eval.py --f './data/eval_results/ASQA/selfrag_reproduction-ASQA-selfrag_llama3_70B-adapter-colbert_api-0614_0303_20/rag_output-selfrag_reproduction|ASQA|selfrag_llama3_70B-adapter|colbert_api|time=0614_0303_20.jsonl' \
    --mauve \
    --qa