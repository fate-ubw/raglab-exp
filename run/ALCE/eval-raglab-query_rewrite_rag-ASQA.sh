# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/query_rewrite_rag-ASQA-Llama3-8B-baseline-colbert_api-0519_1128_04/rag_output-query_rewrite_rag|ASQA|Llama3-8B-baseline|colbert_api|time=0519_1128_04.jsonl' \
    --mauve \
    --qa
