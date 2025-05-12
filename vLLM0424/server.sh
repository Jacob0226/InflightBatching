#!/bin/bash
set -x

# Sample cmd:
#   ./server.sh --model-name meta-llama/Llama-3.1-8B
#   ./server.sh --model-name meta-llama/Llama-3.1-8B --v1

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

# Copy serving_chat.py
cp  $USER/InflightBatching/vLLM0424/serving_chat_0509.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/serving_chat.py

# Function to parse options from any order of arguments
# Function to parse options from any order of arguments
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            --v1)
                V1_FLAG=True 
                shift
                ;;
            -h|--help)
                echo "Usage: $0 --model-name <model_name> "
                echo "  e.g.  ./server.sh --model-name meta-llama/Llama-3.1-8B"
                echo "  --model-name <model_name>  model name"
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
}

# Parse the arguments
V1_FLAG=False 
parse_args "$@"

if [ "$V1_FLAG" = "True" ]; then
    export VLLM_USE_V1=1
    SCHE=0
else
    export VLLM_USE_V1=0
    SCHE=4
fi

MODEL_FOLDER="${MODEL_FOLDER:-/data/huggingface/hub}"
MODEL_PATH="${MODEL_FOLDER}/${MODEL_NAME}"
TP=1
BT=131072 # --max-num-batched-tokens
QUANT=None # --quantization is ONLINE quantization. Cannot use in ServiceNow

if [[ "$MODEL_NAME" == "meta-llama/Llama-3.1-8B" ]]; then
    DTYPE=bfloat16
    KV_TYPE=auto
elif [[ "$MODEL_NAME" == "amd/Llama-3.3-70B-Instruct-FP8-KV" ]]; then
    DTYPE=float16
    if [ "$V1_FLAG" = "True" ]; then # vLLM v1 engine doesn't support kv type in v0.8.3
        KV_TYPE=auto
    else
        KV_TYPE=fp8
    fi
fi

# Locust settings
DURATION=3m # 3 minutes
I_FOLDER=$USER/InflightBatching/Datasets
# For the 70B model, H100 runs with TP=4, so MI300X processes the batch size at one-fourth that of the H100.
# H100 Users=(1 8 16 24 32 40 48 56 64) 
N_USER=(1 2 4 6 8 10 12 14 16) 
I_FILE=(2000.txt 4000.txt 8500.txt)
O_LEN=(200 200 200)
Locust_File=$USER/InflightBatching/vLLM0424/locustfile.py # or locustfile_RandomDataset.py for fully randomized datasets

# Debug usage. Less cases
# DURATION=10s
# N_USER=(1 8)
# O_LEN=(150)
# I_FILE=(2000.txt)

# Output 
# meta-llama/Llama-3.1-8B -> meta-llama_Llama-3.1-8B
out_root_folder="Result/0512/Test"
if [[ "$V1_FLAG" == "True" ]]; then
  result_folder="$out_root_folder/$(basename "$MODEL_NAME")_V1"
else
  result_folder="$out_root_folder/$(basename "$MODEL_NAME")_V0"
fi
mkdir -p $result_folder

SERVER_PORT=8000
Locust_Master_Host=127.0.1.1
Locust_Master_Port=5002

# Command to launch the server
vllm serve $MODEL_PATH \
    --chat-template /app/vllm/examples/tool_chat_template_llama3.1_json.jinja \
    --dtype ${DTYPE} \
    --tensor-parallel-size $TP \
    --kv-cache-dtype ${KV_TYPE} \
    --swap-space 16 \
    --distributed-executor-backend mp \
    --max-num-seqs 64 \
    --max-model-len 16384 \
    --max-seq-len-to-capture 16384 \
    --max-num-batched-tokens $BT \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --disable-log-requests \
    --uvicorn-log-level warning \
    --port $SERVER_PORT \
    $(if [ "$SCHE" -ne 0 ]; then echo "--num-scheduler-steps $SCHE"; fi) &
    # --quantization $QUANT  # error: argument --quantization/-q: invalid choice: 'None'

# Check if the servers are ready
echo "Waiting for the server to be ready on port ${SERVER_PORT}..."
server_url="http://localhost:${SERVER_PORT}"  # Replace with your server's URL and port
server_launch_time=0
# Loop to check server readiness
while ! curl -s "$server_url" > /dev/null; do
    sleep 5  # Wait 5 seconds before checking again
    server_launch_time=$((server_launch_time + 5))
    echo "Server launch time(s): ${server_launch_time}"
done
echo "Server on port ${SERVER_PORT} is ready!"

# Start the servers
for n_user in "${N_USER[@]}"; do # 1
    for idx_file in "${!I_FILE[@]}"; do # i2000.txt
        i_file="${I_FILE[$idx_file]}"
        o_len="${O_LEN[$idx_file]}"
        ilen="${i_file%.txt}" # 2500.txt -> 2500
        n_user_str=$(printf "%02d" "$n_user")user # "8" --> "08"
        target=${result_folder}/${DURATION}_i${ilen}_${n_user_str}

        locust --server vLLM --hf-model $MODEL_PATH \
            -f $Locust_File --host http://localhost:${SERVER_PORT} --endpoint /v1/chat/completions \
            -t $DURATION -u $n_user -r $n_user --processes $n_user \
            -ifile $I_FOLDER/$i_file -olen $o_len \
            -outJson ${result_folder}/metric.json -target $target \
            --master-host $Locust_Master_Host --master-port $Locust_Master_Port --master-bind-port $Locust_Master_Port \
            --headless --only-summary
    done
done

# Kill all vllm server processes
ps -ef | grep '[p]ython' | awk '{print $2}' | xargs kill -9  # Kill server
echo "All servers have been killed."
sleep 10 # Wait a bit before moving on

