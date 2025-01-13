#!/bin/bash
set -x

# Install python lib
pip install --upgrade --ignore-installed blinker
pip install locust orjson xlsxwriter setuptools

# Env var:
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_MIN_NCHANNELS=112
export VLLM_FP8_PADDING=1
export VLLM_FP8_ACT_PADDING=1
export VLLM_FP8_WEIGHT_PADDING=1
export VLLM_FP8_REDUCE_CONV=1
export HIP_FORCE_DEV_KERNARG=1

# Function to parse options from any order of arguments
# Function to parse options from any order of arguments
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            --tp)
                TP="$2"
                shift 2
                ;;
            --instance)
                num_instances="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 --model-name <model_name> --tp <1|2|4|8> --instance <num_instances>"
                echo "  e.g.  ./server.sh --model-name meta-llama/Llama-3.1-8B --tp 4"
                echo "  --model-name <model_name>  Identical to huggingface model name"
                echo "  --tp <1|2|4|8>             Tensor parallel value, must be one of [1, 2, 4, 8]"
                echo "  -h, --help                 Show this help message"
                exit 0
                ;;
            *)  
                echo "Unknown argument: $1"
                exit 1
                ;;
        esac
    done

    if [ -z "$MODEL_NAME" ]; then
        echo "Error: --model <model_name> is required" >&2
        exit 1
    fi

    if [ -z "$num_instances" ]; then
        echo "Error: --instance <num_instances> is required" >&2
        exit 1
    fi

    if [[ ! "$TP" =~ ^(1|2|4|8)$ ]]; then
        echo "Error: --tp must be one of [1, 2, 4, 8]" >&2
        exit 1
    fi
}

# Parse the arguments
parse_args "$@"

MODEL_FOLDER="${MODEL_FOLDER:-/data/huggingface/hub}"
MODEL_PATH="${MODEL_FOLDER}/${MODEL_NAME}"
QUANT=None # --quantization is ONLINE quantization. Cannot use in ServiceNow

if [[ "$MODEL_NAME" == "meta-llama/Llama-3.1-8B" ]]; then
    DTYPE=bfloat16
    KV_TYPE=auto
elif [[ "$MODEL_NAME" == "amd/Llama-3.1-70B-Instruct-FP8-KV" ]]; then
    DTYPE=float16
    if command -v rocm-smi &>/dev/null; then
        KV_TYPE=fp8
    elif command -v nvidia-smi &>/dev/null; then
        KV_TYPE=auto # H100 cannot use fp8
    fi
fi


for i in $(seq 1 $num_instances); do
    SERVER_PORT+=("$(printf '800%1d' $i)")
done

if [ "$num_instances" -eq 8 ]; then
    GPU_Index=(0 1 2 3 4 5 6 7)
elif [ "$num_instances" -eq 4 ]; then
    GPU_Index=("0,1" "2,3" "4,5" "6,7")
elif [ "$num_instances" -eq 2 ]; then
    GPU_Index=("0,1,2,3" "4,5,6,7")
elif [ "$num_instances" -eq 1 ]; then
    GPU_Index=("0,1,2,3,4,5,6,7")
else
    echo "Error: num_instances must be 1, 2, 4 or 8"
    exit 1
fi



# Start the servers
for i in $(seq 0 $(($num_instances - 1))); do
    server_port=${SERVER_PORT[$i]}
    gpu_index=${GPU_Index[$i]}
    
    # Command to launch the server
    CUDA_VISIBLE_DEVICES=${gpu_index} \
    HIP_VISIBLE_DEVICES=${gpu_index} \
    vllm serve $MODEL_PATH \
        --dtype ${DTYPE} \
        --kv-cache-dtype ${KV_TYPE} \
        --quantization $QUANT \
        --swap-space 16 \
        --disable-log-requests \
        --distributed-executor-backend mp \
        --tensor-parallel-size $TP \
        --num-scheduler-steps 16 \
        --enable-chunked-prefill False \
        --max-num-seqs 64 \
        --max-model-len 16384 \
        --max-seq-len-to-capture 16384 \
        --max-num-batched-tokens 131072 \
        --uvicorn-log-level warning \
        --port $server_port &
done

# Check if the servers are ready
for i in $(seq 0 $(($num_instances - 1))); do
    server_port=${SERVER_PORT[$i]}
    echo "Waiting for the server to be ready on port ${server_port}..."
    server_url="http://localhost:${server_port}"  # Replace with your server's URL and port
    server_launch_time=0
    # Loop to check server readiness
    while ! curl -s "$server_url" > /dev/null; do
        sleep 5  # Wait 5 seconds before checking again
        server_launch_time=$((server_launch_time + 5))
        echo "Server launch time(s): ${server_launch_time}"
    done
    echo "Server on port ${server_port} is ready!"
done

echo "All ${num_instances} ${MODEL_NAME}_TP${TP} servers are up and running."

# meta-llama/Llama-3.1-8B -> meta-llama_Llama-3.1-8B
result_folder="$(echo "$MODEL_NAME" | sed 's/\//_/g')_${num_instances}xTP${TP}" 
mkdir -p $result_folder

# Run the client scripts in parallel for all 4 servers
for i in $(seq 0 $(($num_instances - 1))); do
    server_port=${SERVER_PORT[$i]}
    log_folder="${result_folder}/log"
    mkdir -p $log_folder
    logging_file="${log_folder}/client${i}.log"
    ./client.sh --server-idx $i --server-port $server_port --hf-model $MODEL_PATH --result-folder $result_folder \
         2>&1 | tee ${logging_file} &
done

wait

# Sample cmd:
#   ./server.sh --model-name meta-llama/Llama-3.1-8B --tp 1 --instance 8
#   ./server.sh --model-name amd/Llama-3.1-70B-Instruct-FP8-KV --tp 2 --instance 2
#   ./server.sh --model-name amd/Llama-3.1-70B-Instruct-FP8-KV --tp 8 --instance 1