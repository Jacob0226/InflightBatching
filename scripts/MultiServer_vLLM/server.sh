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
    Locust_Master_Host+=("127.0.1.$i")  # Add IP addresses 127.0.1.1, 127.0.1.2, ...
    Locust_Master_Port+=("$(printf '500%1d' $i)") 
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


# Locust settings
DURATION=3m # 3 minutes
I_FOLDER=$USER/POC_RFP/vllm/Llama3.1/Datasets
N_USER=(1 8 16 24 32 40 48 56 64 96 128 196)
I_FILE=(2000.txt 4400.txt 8600.txt)
O_LEN=(150 150 150)
Locust_File=$USER/POC_RFP/vllm/Llama3.1/locustfile.py
model_name="${MODEL_PATH%/*}"  # Remove last component (/data/huggingface/hub/meta-llama/Llama-3.1-8B  --> Llama-3.1-8B)
model_name="${model_name##*/}/$(basename "$MODEL_PATH")"  # Extract last two components

# Debug usage. Less cases
# DURATION=10s
# N_USER=(1 8)
# O_LEN=(150)
# I_FILE=(2000.txt)

# Output 
# meta-llama/Llama-3.1-8B -> meta-llama_Llama-3.1-8B
result_folder="0331_${MODEL_NAME//\//_}_${num_instances}xTP${TP}"
log_folder="${result_folder}/log"
benchmark_folder="${result_folder}/LocustMetric"
openai_metric_folder="${result_folder}/OpenAI_Metric"
mkdir -p $log_folder $benchmark_folder $openai_metric_folder


# Start the servers
for n_user in "${N_USER[@]}"; do
    for idx_file in "${!I_FILE[@]}"; do
        i_file="${I_FILE[$idx_file]}"
        o_len="${O_LEN[$idx_file]}"
        ilen="${i_file%.txt}" # 2500.txt -> 2500
        n_user_str=$(printf "%02d" "$n_user")user # "8" --> "08"

        # Start servers in parallel
        for idx_server in $(seq 0 $(($num_instances - 1))); do
            server_port=${SERVER_PORT[$idx_server]}
            gpu_index=${GPU_Index[$idx_server]}
            
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
                --enable-chunked-prefill False \
                --max-num-seqs 64 \
                --max-model-len 16384 \
                --max-seq-len-to-capture 16384 \
                --max-num-batched-tokens 131072 \
                --uvicorn-log-level warning \
                --port $server_port &

            # Save the server PID (to kill later if needed)
            server_pids+=($!) 
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

        # Run locust in parallel for all servers
        declare -A locust_pids  # Associative array to map idx_server to PID

        for idx_server in $(seq 0 $(($num_instances - 1))); do
            server_port=${SERVER_PORT[$idx_server]}
            Master_Host=${Locust_Master_Host[$idx_server]}
            Master_Port=${Locust_Master_Port[$idx_server]}

            logging_file="${log_folder}/locust_server${idx_server}.log"
            target=${result_folder}/${DURATION}_i${ilen}_${n_user_str}
            
            # Start locust in the background and save the PID
            locust --server vLLM --hf-model $MODEL_PATH \
                -f $Locust_File --host http://localhost:${server_port} --endpoint /v1/completions \
                -t $DURATION -u $n_user -r $n_user --processes 16 \
                -ifile $I_FOLDER/$i_file -olen $o_len \
                -outJson ${benchmark_folder}/server${idx_server}.json -target $target \
                --master-host $Master_Host --master-port $Master_Port --master-bind-port $Master_Port \
                --headless --only-summary 2>&1 | tee ${logging_file} &

            # Map idx_server to the PID of the locust process
            locust_pids["$idx_server"]=$! 
        done

        # Wait for all locust processes to finish
        for idx in "${!locust_pids[@]}"; do
            pid="${locust_pids[$idx]}"
            server_port="${SERVER_PORT[$idx]}"
            idx_server=$idx

            wait "$pid"
            echo "Locust process for server index $idx_server (PID: $pid) finishes..."
            curl "http://localhost:${server_port}/metrics" > \
                "$openai_metric_folder/${DURATION}_i${ilen}_${n_user_str}_server${idx_server}.log"
        done

        echo "All locust tasks are complete."

        # Kill all vllm server processes
        ps -ef | grep '[p]ython' | awk '{print $2}' | xargs kill -9  # Kill server
        # for pid in "${server_pids[@]}"; do # Not work. Server still alive
        #     kill -9 $pid
        # done

        echo "All servers have been killed."
        
        sleep 10 # Wait a bit before moving on
    done
done


# Sample cmd:
#   ./server.sh --model-name meta-llama/Llama-3.1-8B --tp 1 --instance 8
#   ./server.sh --model-name amd/Llama-3.1-70B-Instruct-FP8-KV --tp 2 --instance 2
#   ./server.sh --model-name amd/Llama-3.1-70B-Instruct-FP8-KV --tp 8 --instance 1