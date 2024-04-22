export CUDA_VISIBLE_DEVICES=6
python -i /home/wyd/raglab-exp/ALCE/eval.py --f '/home/wyd/raglab-exp/1-eval_output/ASQA/rag_output-algorithm_name=naive_rag|task=ASQA|llm_path=Llama-2-7b-chat-hf|temperature=0.0|top_p=1.0|generation_stop=|generate_maxlength=300|n_docs=10|time:0421_0311.jsonl' \
    --mauve \
    --qa
