#!/bin/bash
set -x

# TensorRT-LLM/examples/quantization/quantize.py quantize 70B-bf16 into fp8 and it's already TRT-LLM ckpt 
MODEL_NAME=(
    "meta-llama/Llama-3.1-8B 1 bfloat16" 
    "meta-llama/Llama-3.1-8B 2 bfloat16" 
    "meta-llama/Llama-3.1-70B 4 fp8"
    "meta-llama/Llama-3.1-70B 8 fp8"
    )
MODEL_PATH=/data/huggingface/hub/
CONVERT_SCRIPT=/home/jacchang/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py
QUANT_SCRIPT=/home/jacchang/tensorrtllm_backend/tensorrt_llm/examples/quantization/quantize.py

# Generate TRT-LLM ckpt 
for target in "${MODEL_NAME[@]}"; do
    read model_name tp dtype <<< "$target"

    # Check for dtype being 'fp8' or not
    if [[ "$dtype" != "fp8" ]]; then
        python3 $CONVERT_SCRIPT \
            --model_dir ${MODEL_PATH}/${model_name} \
            --output_dir ${MODEL_PATH}/tllm_checkpoint/${model_name}_${dtype}_${tp}GPU \
            --dtype ${dtype} \
            --tp_size ${tp}
    else
        python3 $QUANT_SCRIPT \
            --model_dir ${MODEL_PATH}/${model_name} \
            --qformat fp8 --kv_cache_dtype fp8 --tp_size ${tp} \
            --output_dir ${MODEL_PATH}/tllm_checkpoint/${model_name}_${dtype}_${tp}GPU
    fi

    echo "Model checkpoint for ${model_name} with dtype ${dtype} and TP size ${tp} GPU(s) generated."

done

# Build TRT-LLM engine
for target in "${MODEL_NAME[@]}"; do
    read model_name tp dtype <<< "$target"

    # vLLM: --max-model-len 16384 --max-num-batched-tokens 131072 --max-seq-len-to-capture 16384										
    trtllm-build \
        --checkpoint_dir ${MODEL_PATH}/tllm_checkpoint/${model_name}_${dtype}_${tp}GPU \
        --output_dir $MODEL_PATH/trt_engines/${model_name}_${dtype}_${tp}GPU \
        --gemm_plugin ${dtype} \
        --max_batch_size 64  \
        --max_input_len  15384 \
        --max_seq_len    16384 \
        --max_num_tokens 131072 \
        --workers ${tp}

    echo "------------------------------------------------------------"
done    


