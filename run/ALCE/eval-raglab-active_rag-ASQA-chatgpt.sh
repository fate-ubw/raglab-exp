# export CUDA_VISIBLE_DEVICES=6
python  ./ALCE/eval.py --f './data/eval_results/ASQA/active_rag-ASQA-Llama3-8B-baseline-colbert_api-0520_0032_05/rag_output-active_rag|ASQA|Llama3-8B-baseline|colbert_api|time=0520_0032_05.jsonl' \
    --mauve \
    --qa