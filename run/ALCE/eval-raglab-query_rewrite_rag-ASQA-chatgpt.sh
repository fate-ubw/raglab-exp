# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/query_rewrite_rag-ASQA-gpt-3.5-turbo-colbert_api-0522_1138_28/rag_output-query_rewrite_rag|ASQA|gpt-3.5-turbo|colbert_api|time=0522_1138_28.jsonl' \
    --mauve \
    --qa
