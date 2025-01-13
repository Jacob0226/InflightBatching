# This script is called by server_8B_bf16_TP2.sh
#!/bin/bash
set -x

# Function to parse options from any order of arguments
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --server-idx)
                SERVER_IDX="$2"
                shift 2
                ;;
            --server-port)
                SERVER_PORT="$2"
                shift 2
                ;;
            --hf-model)
                MODEL_PATH="$2"
                shift 2
                ;;
            --result-folder)
                result_folder="$2"
                shift 2
                ;;
            *)  
                echo "Unknown argument: $1"
                exit 1
                ;;
        esac
    done
}

# Parse the arguments
parse_args "$@"

# Checking if necessary arguments are provided
if [ -z "$SERVER_IDX" ]; then
    echo "Error: server-idx is required" >&2
    exit 1
fi

if [ -z "$SERVER_PORT" ]; then
    echo "Error: server-port is required" >&2
    exit 1
fi

if [ -z "$MODEL_PATH" ]; then
    echo "Error: hf-model is required" >&2
    exit 1
fi

if [ -z "$result_folder" ]; then
    echo "Error: result_folder is required" >&2
    exit 1
fi

# Run locust with the server URL and model path
echo "Starting locust client for number#${SERVER_IDX} server at port $SERVER_PORT ..."
echo "The serving model is $MODEL_PATH ..."

for i in $(seq 1 8); do
    Locust_Master_Host+=("127.0.1.$i")  # Add IP addresses 127.0.1.1, 127.0.1.2, ...
    Locust_Master_Port+=("$(printf '500%1d' $i)") 
done
Master_Host=${Locust_Master_Host[$SERVER_IDX]}
Master_Port=${Locust_Master_Port[$SERVER_IDX]}

DURATION=3m # 3 minutes
N_USER=(1 8 16 24 32 40 48 56 64)
I_FOLDER=/home/jacchang/POC_RFP/vllm/Llama3.1/Datasets
I_FILE=(2500.txt 5500.txt 11000.txt)
Locust_File=/home/jacchang/POC_RFP/vllm/Llama3.1/locustfile.py
model_name="${MODEL_PATH%/*}"  # Remove last component (/data/huggingface/hub/meta-llama/Llama-3.1-8B  --> Llama-3.1-8B)
model_name="${model_name##*/}/$(basename "$MODEL_PATH")"  # Extract last two components
# tp=$(echo $result_folder | sed 's/.*_TP\([0-9]*\)/\1/') # get TP=1 from meta-llama_Llama-3.1-8B_TP1 


# Debug usage. Less cases
# DURATION=5s
# N_USER=(128)
# I_FILE=(11000.txt)

benchmark_folder="${result_folder}/benchmark"
mkdir -p $benchmark_folder
for n_user in "${N_USER[@]}"; do
    for i_file in "${I_FILE[@]}"; do
        ilen="${i_file%.txt}" # 2500.txt -> 2500
        n_user_str=$(printf "%02d" "$n_user")user # "8" --> "08"
        
        target=${result_folder}/${DURATION}_i${ilen}_${n_user_str}

        locust --server vLLM --hf-model $MODEL_PATH \
            -f $Locust_File --host http://localhost:${SERVER_PORT} --endpoint /v1/completions \
            -t $DURATION -u $n_user -r $n_user --processes 16 \
            -ifile $I_FOLDER/$i_file -olen 350 \
            -outJson ${benchmark_folder}/server${SERVER_IDX}.json -target $target \
            --master-host $Master_Host --master-port $Master_Port --master-bind-port $Master_Port \
            --headless --only-summary 

        echo -e "------------------------------------------------------------\n" 

    done
done

sleep 10 # Wait other parallel models before killing the server process
ps -ef | grep '[p]ython' | awk '{print $2}' | xargs kill -9  # Kill server
sleep 10 

echo -e "----------------------End of Server--------------------\n\n"


