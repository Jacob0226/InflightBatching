N_GPU=(2 4)
N_PARAM=(8 70)

for n_gpu in "${N_GPU[@]}"; do
    for n_param in "${N_PARAM[@]}"; do
        ENGINE_DIR=/data/huggingface/hub/meta-llama/trt_engines/Llama-3.1-${n_param}B_float16_${n_gpu}GPU/
        TOKENIZER_DIR=/data/huggingface/hub/meta-llama/Llama-3.1-${n_param}B
        MODEL_FOLDER=/home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_llama-3.1-${n_param}b_float16_${n_gpu}GPU
        FILL_TEMPLATE_SCRIPT=/home/jacchang/tensorrtllm_backend/tools/fill_template.py
        TRITON_MAX_BATCH_SIZE=64
        INSTANCE_COUNT=1
        MAX_QUEUE_DELAY_MS=0
        MAX_QUEUE_SIZE=0
        DECOUPLED_MODE=true # setting the stream tensor to true

        python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}
        python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
        python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE}
        python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT},max_queue_size:${MAX_QUEUE_SIZE}
        python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT}


    done
done    
