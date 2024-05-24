# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/self_ask-ASQA-gpt-3.5-turbo-colbert_api-0522_2008_13/rag_output-self_ask|ASQA|gpt-3.5-turbo|colbert_api|time=0522_2008_13.jsonl' \
    --mauve \
    --qa
