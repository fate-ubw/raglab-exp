# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/naive_rag-ASQA-Llama3-8B-baseline-colbert-0519_0948_16/rag_output-naive_rag|ASQA|Llama3-8B-baseline|colbert|time=0519_0948_16.jsonl' \
    --mauve \
    --qa