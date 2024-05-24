# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/naive_rag-ASQA-Llama3-8B-baseline-colbert_api-0519_1007_20/rag_output-naive_rag|ASQA|Llama3-8B-baseline|colbert_api|time=0519_1007_20.jsonl' \
    --mauve \
    --qa
