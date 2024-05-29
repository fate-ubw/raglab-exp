# export CUDA_VISIBLE_DEVICES=6
python  ./ALCE/eval.py --f './data/eval_results/ASQA/iter_retgen-ASQA-Llama3-8B-baseline-colbert_api-0528_0253_30/rag_output-iter_retgen|ASQA|Llama3-8B-baseline|colbert_api|time=0528_0253_30.jsonl' \
    --mauve \
    --qa