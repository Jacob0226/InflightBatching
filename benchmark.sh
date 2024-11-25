#!/bin/bash

# Sample cmd:
# ./benchmark.sh  --server vLLM
# ./benchmark.sh  --server Triton

function show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help                  Show this help message and exit"
    echo "  -o                          Output folder"
    echo "  --server ServerType         Choices=[vLLM, Triton]. Note: Triton is Triton-Inference-Server"
    echo "E.g., "
    echo "  ./benchmark_scan.sh --server vLLM"
    echo
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --server)
            if [[ "$2" == "vLLM" || "$2" == "Triton" ]]; then
                server=$2
                shift 2
            else
                echo "Error: --server must be either 'vLLM' or 'Triton'."
                exit 1
            fi
            ;;
        --model)
            if [[ -n "$2" ]]; then
                model_name=$2
                shift 2
            else
                echo "Error: --model requires a value."
                exit 1
            fi
            ;;
        --tp)
            if [[ -n "$2" ]]; then
                tp=$2
                shift 2
            else
                echo "Error: --tp requires a value."
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done


if [ "${server}" == "vLLM" ]; then
    endpoint=/v1/completions
elif [ "${server}" == "Triton" ]; then
    endpoint=/v2/models/ensemble/generate_stream
fi



# Env var:
MODEL_FOLDER=/data/huggingface/hub/meta-llama/
MODEL_FOLDER=/data/huggingface/hub/amd/
DTYPE=(float16) # fp8
DURATION=3m # 3 minutes
N_USER=(1 8 16 24 32 40 48 56 64) # (1 32 64)  
I_FOLDER=Datasets
I_FILE=(2500.txt 5500.txt 11000.txt)

# # Debug usage. Less cases
# DURATION=30s
# N_USER=(1)
# I_FILE=(2500.txt)

HF_MODEL_PATH="${MODEL_FOLDER}${model_name}"

if { [ "$model_name" == "Llama-3.1-8B" ] || 
     [ "$model_name" == "Llama-3.1-8B-Instruct-FP8-KV" ]; } && 
     [ "$tp" -eq 1 ]; then
    GPU_Index="1"
    SERVER_PORT=8000
    RPC_PORT=2000
    Master_Host=127.0.0.1
    Master_Port=5557
elif { [ "$model_name" == "Llama-3.1-8B" ] || 
     [ "$model_name" == "Llama-3.1-8B-Instruct-FP8-KV" ]; } && 
     [ "$tp" -eq 2 ]; then
    GPU_Index="2,3"
    SERVER_PORT=8001
    RPC_PORT=2001
    Master_Host=127.0.0.2
    Master_Port=5558 
elif [[ "$model_name" == "Llama-3.1-70B" || "$model_name" == "Llama-3.1-70B-Instruct-FP8-KV" ]]; then
    GPU_Index="4,5,6,7"
    SERVER_PORT=8002
    RPC_PORT=2002
    Master_Host=127.0.0.3
    Master_Port=5559
fi
logging_file="benchmark_${model_name}_TP${tp}.log"
rm $logging_file

if [ "${server}" == "vLLM" ]; then
    if [ "${model_name}" == "Llama-3.1-8B" ]; then
        max_num_batched_tokens=300000
    elif [ "${model_name}" == "Llama-3.1-70B" ]; then
        max_num_batched_tokens=131072
    fi

    if [ "${model_name}" == "Llama-3.1-8B-Instruct-FP8-KV" ]; then
        max_num_batched_tokens=300000
    elif [ "${model_name}" == "Llama-3.1-70B-Instruct-FP8-KV" ]; then
        max_num_batched_tokens=131072
    fi
    # --enable-prefix-caching \
    # --distributed-executor-backend ray \
    # --dtype float16 \
    # export VLLM_LOGGING_LEVEL=WARNING 
    server_cmd="
        HIP_VISIBLE_DEVICES=${GPU_Index} \
        vllm serve $HF_MODEL_PATH \
            --swap-space 16 \
            --disable-log-requests \
            --tensor-parallel-size $tp \
            --num-scheduler-steps 10 \
            --enable-chunked-prefill False \
            --max-num-seqs 64  \
            --max-model-len 11500 \
            --max-seq-len-to-capture 11500 \
            --max-num-batched-tokens $max_num_batched_tokens \
            --uvicorn-log-level warning \
            --port $SERVER_PORT \
            2>&1 | tee -a ${logging_file} &
    "
elif [ "${server}" == "Triton" ]; then
    model_lowercase=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
    Triton_MODEL_FOLDER=/home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_${model_lowercase}_${DTYPE}_${tp}GPU
    server_cmd="
        CUDA_VISIBLE_DEVICES=${GPU_Index} \
        python3 /home/jacchang/tensorrtllm_backend/scripts/launch_triton_server.py \
            --world_size=$tp  --model_repo=${Triton_MODEL_FOLDER} 
    "
fi
echo "server_cmd=${server_cmd}" 2>&1 | tee -a "${logging_file}"
eval "${server_cmd}"

# Check if the server is ready
echo "Waiting for the server to be ready..."
server_url="http://localhost:${SERVER_PORT}"  # Replace with your server's URL and port
server_launch_time=0
# Loop to check server readiness
while ! curl -s "$server_url" > /dev/null; do
    sleep 5  # Wait 5 second before checking again
    server_launch_time=$((server_launch_time + 5))
    echo "${model_name}_TP${tp} Server launch time(s): ${server_launch_time}"
done


for n_user in "${N_USER[@]}"; do
    for i_file in "${I_FILE[@]}"; do
        ilen="${i_file%.txt}" # 2500.txt -> 2500
        n_user_str=$(printf "%02d" "$n_user")user # "8" --> "08"
        target=${model_name}_${DTYPE}_TP${tp}/${DURATION}/i${ilen}/${n_user_str}

        locust_cmd="
            locust --server $server --hf-model $HF_MODEL_PATH \
                --host http://localhost:${SERVER_PORT} --endpoint $endpoint \
                -t $DURATION -u $n_user -r $n_user --processes 8 \
                -ifile $I_FOLDER/$i_file -olen 350 -target $target \
                --master-host $Master_Host --master-port $Master_Port --master-bind-port $Master_Port \
                --headless --only-summary \
                2>&1 | tee -a ${logging_file}
        "

        echo "locust_cmd=${locust_cmd} \n" 2>&1 | tee -a "${logging_file}"
        eval "${locust_cmd}"
        echo -e "------------------------------------------------------------\n" | tee -a "${logging_file}"

    done
done

sleep 120 # Wait other parallel models before killing the server process
ps -ef | grep '[p]ython' | awk '{print $2}' | xargs kill -9  # Kill server
pkill tritonserver    

echo -e "----------------------End of Server--------------------\n\n" | tee -a "${logging_file}"


