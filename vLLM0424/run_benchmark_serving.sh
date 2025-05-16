#!/bin/bash
set -x

# Sample cmd:
#   ./run_benchmark_serving.sh --model-name meta-llama/Llama-3.1-8B
#   ./run_benchmark_serving.sh --model-name amd/Llama-3.3-70B-Instruct-FP8-KV

# Env var:
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_MIN_NCHANNELS=112
export VLLM_FP8_PADDING=1
export VLLM_FP8_ACT_PADDING=1
export VLLM_FP8_WEIGHT_PADDING=1
export VLLM_FP8_REDUCE_CONV=1
export HIP_FORCE_DEV_KERNARG=1

export VLLM_USE_V1=1

# Function to parse options from any order of arguments
# Function to parse options from any order of arguments
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --model-name)
                MODEL_NAME="$2"
                shift 2
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
parse_args "$@"


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
    # Not sure if this docker (rocm/vllm-dev:nightly_rc2.2_perf_revert_test_20250513) support fp8 in v1 engine
    KV_TYPE=fp8 
fi


# Output 
# meta-llama/Llama-3.1-8B -> meta-llama_Llama-3.1-8B
result_folder="Result/0516/DiffSeed/$(basename "$MODEL_NAME")"
mkdir -p $result_folder

SERVER_PORT=8000
Locust_Master_Host=127.0.1.1
Locust_Master_Port=5002

# Command to launch the server
server_log=$result_folder/server_log.txt
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
    --enable-prefix-caching \
    --no-enable-chunked-prefill \
    --disable-log-requests \
    --uvicorn-log-level warning \
    --port $SERVER_PORT  &
    # 2>&1 | tee "${server_log}" 
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

CONCURRENCY=(1 2 4 6 8 10 12 14 16)
ILEN=(2000 4000 8500)
olen=200
seed=0

# Warmup
python3 /app/vllm/benchmarks/benchmark_serving.py --host localhost --backend openai --port $SERVER_PORT \
    --model $MODEL_PATH --dataset-name random --num-prompts 100 --ignore-eos \
    --random-input-len 128 --random-output-len 128 --random-range-ratio 0 --seed $seed \
    --max-concurrency 64 --percentile-metrics ttft,tpot,itl,e2el

for concurrency in "${CONCURRENCY[@]}"; do
    for ilen in "${ILEN[@]}"; do
        num_prompts=$((concurrency * 20))
        seed=$((seed + 1))
        # Define the benchmark file path
        benchmark_file="${result_folder}/i${ilen}_o${olen}_c${concurrency}_p${num_prompts}.log"

        # Run the benchmark and capture the output in a log file
        python3 /app/vllm/benchmarks/benchmark_serving.py \
            --host localhost \
            --backend openai \
            --port "$SERVER_PORT" \
            --model $MODEL_PATH \
            --dataset-name random \
            --num-prompts $num_prompts \
            --random-input-len "$ilen" \
            --random-output-len "$olen" \
            --random-range-ratio 0 \
            --seed $seed \
            --max-concurrency "$concurrency" \
            --percentile-metrics ttft,tpot,itl,e2el \
            2>&1 | tee "${benchmark_file}"
    done  
done


# Kill all vllm server processes
ps -ef | grep '[p]ython' | awk '{print $2}' | xargs kill -9  # Kill server
echo "All servers have been killed."
sleep 10 # Wait a bit before moving on

