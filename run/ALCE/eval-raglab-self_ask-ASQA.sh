# export CUDA_VISIBLE_DEVICES=0
python  ./ALCE/eval.py --f './data/eval_results/ASQA/self_ask-ASQA-Llama3-8B-baseline-colbert_api-0519_1902_44/rag_output-self_ask|ASQA|Llama3-8B-baseline|colbert_api|time=0519_1902_44.jsonl' \
    --mauve \
    --qa
