#!/bin/bash
set -x

MODEL_NAME=(
    # Llama-3.1-8B_bfloat16_TP1
    "/data/huggingface/hub/trt_engines/meta-llama/Llama-3.1-8B_bfloat16_1GPU" \
    "/data/huggingface/hub/meta-llama/Llama-3.1-8B" \
    "/home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-8B_bfloat16_1GPU"

    # Llama-3.1-8B_bfloat16_TP2
    "/data/huggingface/hub/trt_engines/meta-llama/Llama-3.1-8B_bfloat16_2GPU" \
    "/data/huggingface/hub/meta-llama/Llama-3.1-8B" \
    "/home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-8B_bfloat16_2GPU"
    
    # Llama-3.1-70B_fp8_TP4
    "/data/huggingface/hub/trt_engines/meta-llama/Llama-3.1-70B_fp8_4GPU" \
    "/data/huggingface/hub/meta-llama/Llama-3.1-70B" \
    "/home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-70B_fp8_4GPU"

    # Llama-3.1-70B_fp8_TP8
    "/data/huggingface/hub/trt_engines/meta-llama/Llama-3.1-70B_fp8_8GPU" \
    "/data/huggingface/hub/meta-llama/Llama-3.1-70B" \
    "/home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-70B_fp8_8GPU"
)

for target in "${MODEL_NAME[@]}"; do
    if [ -z "$engine_dir" ]; then
        engine_dir=$target
    elif [ -z "$token_dir" ]; then
        token_dir=$target
    elif [ -z "$model_dir" ]; then
        model_dir=$target

        ENGINE_DIR=$engine_dir
        TOKENIZER_DIR=$token_dir
        MODEL_FOLDER=$model_dir
        FILL_TEMPLATE_SCRIPT=/home/jacchang/tensorrtllm_backend/tools/fill_template.py
        TRITON_MAX_BATCH_SIZE=64
        INSTANCE_COUNT=1
        MAX_QUEUE_DELAY_MS=0
        MAX_QUEUE_SIZE=0
        DECOUPLED_MODE=true # setting the stream tensor to true

        # echo "Processing: engine_dir=$engine_dir, token_dir=$token_dir, model_dir=$model_dir"
        # echo "--------------"


        python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}
        python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
        python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE}
        python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT},max_queue_size:${MAX_QUEUE_SIZE}
        python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT}

        # Reset for next set
        engine_dir=""
        token_dir=""
        model_dir=""
    fi
done    
