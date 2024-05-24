# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/naive_rag-ASQA-gpt-3.5-turbo-colbert_api-0522_0931_54/rag_output-naive_rag|ASQA|gpt-3.5-turbo|colbert_api|time=0522_0931_54.jsonl' \
    --mauve \
    --qa
