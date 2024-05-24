# export CUDA_VISIBLE_DEVICES=6
python  ./ALCE/eval.py --f './data/eval_results/ASQA/iter_retgen-ASQA-Llama3-8B-baseline-colbert_api-0519_1421_07/rag_output-iter_retgen|ASQA|Llama3-8B-baseline|colbert_api|time=0519_1421_07.jsonl' \
    --mauve \
    --qa