#!/bin/bash


MODEL_NAME=(Llama-3.1-8B Llama-3.1-70B)
MODEL_PATH=/data/huggingface/hub/meta-llama
CONVERT_SCRIPT=/home/jacchang/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py
DTYPE=float16 #(float16 float8)
N_GPU=(2 4)

# Generate TRT-LLM ckpt 
for model_name in "${MODEL_NAME[@]}"; do
    for n_gpu in "${N_GPU[@]}"; do

        cmd="
            python3 $CONVERT_SCRIPT \
                --model_dir ${MODEL_PATH}/${model_name} \
                --output_dir ${MODEL_PATH}/tllm_checkpoint/${model_name}_${DTYPE}_${n_gpu}GPU \
                --dtype ${DTYPE} \
                --tp_size ${n_gpu}
        "

        echo $cmd
        eval $cmd
        echo "------------------------------------------------------------"
    done
done

# Build TRT-LLM engine
for model_name in "${MODEL_NAME[@]}"; do
    for n_gpu in "${N_GPU[@]}"; do

        cmd="
            CUDA_VISIBLE_DEVICES=4,5,6,7 \
            trtllm-build \
                --checkpoint_dir ${MODEL_PATH}/tllm_checkpoint/${model_name}_${DTYPE}_${n_gpu}GPU \
                --output_dir $MODEL_PATH/trt_engines/${model_name}_${DTYPE}_${n_gpu}GPU \
                --gemm_plugin float16 \
                --max_batch_size 64  \
                --max_input_len  11000 \
                --max_seq_len    11500 \
                --max_num_tokens 704000 \
                --workers ${n_gpu}
        "

        echo $cmd
        eval $cmd
        echo "------------------------------------------------------------"

    done
done    


