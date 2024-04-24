export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file /workspace/self-rag/retrieval_lm/stage3_no_offloading_accelerate.conf \
    /workspace/raglab-exp/raglab/rag/train_alg/finetune.py \
    --model_name_or_path /workspace/raglab-exp/model/Llama-2-7b-chat-hf\
    --use_flash_attn \
    --tokenizer_name /workspace/raglab-exp/model/Llama-2-7b-chat-hf \
    --use_slow_tokenizer \
    --train_file /workspace/train_data/full_output_1005-10samples.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir /workspace/self-rag/output/self_rag_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens \
    