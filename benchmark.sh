#!/bin/bash

# Sample cmd:
# ./benchmark.sh -o report_1107 --server vLLM
# ./benchmark.sh -o report_1107 --server Triton

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
        -o)
            root_out_folder=$2
            shift 2
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
MODEL_NAME=(Llama-3.1-8B Llama-3.1-70B)
TP=(2 4)
SERVER_PORT=8000
DTYPE=float16
DURATION=2m # 2 minutes
N_USER=(1 32 64) # (1 4 8 16 24 32 40 48 56 64)
I_FOLDER=Datasets
I_FILE=(2500.txt 5500.txt 11000.txt)
logging_file=benchmark.log
rm $logging_file

# Debug usage. Less cases
# DURATION=10s # 2 minutes
# N_USER=(1)
# I_FILE=(2500.txt)
# TP=(4)

for model_name in "${MODEL_NAME[@]}"; do
    for tp in "${TP[@]}"; do
        HF_MODEL_PATH="${MODEL_FOLDER}${model_name}"
        echo "HF_MODEL_PATH = ${HF_MODEL_PATH}"
        server_cmd="
            HIP_VISIBLE_DEVICES=4,5,6,7 \
            vllm serve $HF_MODEL_PATH \
                --dtype float16 \
                --distributed-executor-backend ray \
                --swap-space 16 \
                --disable-log-requests \
                --tensor-parallel-size $tp \
                --num-scheduler-steps 10 \
                --enable-prefix-caching \
                --enable-chunked-prefill False \
                --max-num-seqs 64  \
                --max-model-len 11500 \
                --max-num-batched-tokens 704000 \
                --max-seq-len-to-capture 11500 \
                --port $SERVER_PORT \
                2>&1 | tee -a ${logging_file} &
        "
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
            echo "Server launch time(s): ${server_launch_time}"
        done

        for n_user in "${N_USER[@]}"; do
            for i_file in "${I_FILE[@]}"; do
                ilen="${i_file%.txt}" # 2500.txt -> 2500
                n_user_str=$(printf "%02d" "$n_user")user # "8" --> "08"
                o_folder=${root_out_folder}/${model_name}_${DTYPE}_TP${tp}/${DURATION}/i${ilen}/${n_user_str}
   
                locust_cmd="
                    locust --server $server --hf-model $HF_MODEL_PATH \
                        --host http://localhost:${SERVER_PORT} --endpoint $endpoint \
                        -t $DURATION -u $n_user -r $n_user \
                        -ifile $I_FOLDER/$i_file -olen 350 -out $o_folder \
                        --headless --only-summary --csv $o_folder/report --html $o_folder/report.html \
                        2>&1 | tee -a ${logging_file}
                "

                echo "locust_cmd=${locust_cmd}" 2>&1 | tee -a "${logging_file}"
                eval "${locust_cmd}"
                echo -e "------------------------------------------------------------\n" | tee -a "${logging_file}"

            done
        done
        ps -ef | grep '[p]ython' | awk '{print $2}' | xargs kill -9  # Kill server
        echo -e "----------------------End of Server--------------------\n\n" | tee -a "${logging_file}"
    done
done

