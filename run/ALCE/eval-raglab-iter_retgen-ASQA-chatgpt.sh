# export CUDA_VISIBLE_DEVICES=6
python  ./ALCE/eval.py --f './data/eval_results/ASQA/iter_retgen-ASQA-gpt-3.5-turbo-colbert_api-0522_1437_52/rag_output-iter_retgen|ASQA|gpt-3.5-turbo|colbert_api|time=0522_1437_52.jsonl' \
    --mauve \
    --qa