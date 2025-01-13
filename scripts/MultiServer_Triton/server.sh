#!/bin/bash
set -x



# Env var:
tensorrtllm_backend_folder="${TENSORRTLLM_BACKEND_FOLDER:-/home/jacchang/tensorrtllm_backend/}"
STARTING_PORT=7000
export NCCL_MIN_NCHANNELS=112

# Function to parse options from any order of arguments
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --model-repo)
                MODEL_REPOSITORY="$2"
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
                echo "Usage: $0 --model-repo <model_repo> --tp <1|2|4|8> --instance <num_instances>"
                echo "  e.g.  ./server.sh --model-repo /home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-8B_bfloat16_2GPU --tp 2 --instance 4"
                echo "  --model-repo <model_repo>  Path to the Triton Model Repository"
                echo "  --tp <TP>             Tensor parallel value. It must match the setting when TRTLLM builds the engine"
                echo "  --instance <num_instances>  Number of instances to launch"
                echo "  -h, --help                 Show this help message"
                exit 0
                ;;
            *)  
                echo "Unknown argument: $1"
                exit 1
                ;;
        esac
    done

    if [ -z "$MODEL_REPOSITORY" ]; then
        echo "Error: --model-repo <model_repo> is required" >&2
        exit 1
    fi

    if [ ! -d "$MODEL_REPOSITORY" ]; then
        echo "Model repository directory does not exist: $MODEL_REPOSITORY"
        exit 1
    fi

    if [[ -z "$TP" ]]; then
        echo "Error: --tp <TP> is required" >&2
        exit 1
    fi

    if [[ -z "$num_instances" ]]; then
        echo "Error: --instance <num_instances> is required" >&2
        exit 1
    fi

    # Check if TP and num_instances satisfy the condition
    if ! [[ "$TP" =~ ^[0-9]+$ ]] || ! [[ "$num_instances" =~ ^[0-9]+$ ]] || [ $((TP * num_instances)) -gt 8 ]; then
        echo "Error: TP multiplied by the number of instances must be less than or equal to 8" >&2
        exit 1
    fi
}

# Parse the arguments
parse_args "$@"

# Assign GPU index based on the number of instances
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

declare -a container_names
declare -a server_ports

# Start container
for (( i=1; i<=$num_instances; i++ )) do
    container_name="$(whoami)_triton_server_$i"
    container_names+=($container_name)
    
    # Store ports for each container in an array
    port_base=$((STARTING_PORT + (i-1)*3))
    server_ports+=($port_base)
    gpu_index="${GPU_Index[$i-1]}"
    
    # Command to launch the server
    docker run -t -d --gpus all --name $container_name --shm-size=2g \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -p $port_base:8000 \
        -p $((port_base + 1)):8001 \
        -p $((port_base + 2)):8002 \
        -v ${tensorrtllm_backend_folder}:/tensorrtllm_backend \
        -v ~:/home/jacchang -v /scratch1:/data \
        -e CUDA_VISIBLE_DEVICES=$gpu_index \
        nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3 

    docker logs -f $container_name &
done

# Launch server 
for (( i=1; i<=$num_instances; i++ )) do
    container_name="${container_names[$i-1]}"
    # Weird! If the script doesn't use "> /triton_launch.log 2>&1", triton_server is killed after it is init.
    docker exec $container_name bash -c \
        "python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=${TP} \
        --model_repo=${MODEL_REPOSITORY} > /triton_launch.log 2>&1"
done

# Check if the servers are ready
for server_port in "${server_ports[@]}"; do
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

echo "All ${num_instances} of ${MODEL_REPOSITORY} servers are up and running."

# Generate a result folder name based on the model repo, num_instances, and TP
result_folder="$(pwd)/$(basename "$MODEL_REPOSITORY")_${num_instances}xTP${TP}"
log_folder="${result_folder}/log"
rm -rf $result_folder
mkdir -p $result_folder
mkdir -p $log_folder

pids=()
# Run the client scripts in parallel
for (( i=1; i<=$num_instances; i++ )) do
    container_name="${container_names[$i-1]}"
    server_port=${server_ports[$((i-1))]}  # Get the corresponding server port
    logging_file="${log_folder}/client${i}.log"
    
    # Run client.sh inside the respective container
    docker exec $container_name bash -c \
        "pip install --upgrade --ignore-installed blinker && \
        pip install locust orjson xlsxwriter && \
        $(pwd)/client.sh --server-idx $i --server-port $server_port --result-folder $result_folder 2>&1 | tee ${logging_file}" &
    
    # docker logs -f $container_name &
    pids+=($!)
done

# Wait for all locust jobs to finish
for pid in "${pids[@]}"; do
    wait $pid
done

# After tests are completed, stop and remove all containers
echo "Stopping and removing all containers..."
for container_name in "${container_names[@]}"; do
    docker stop $container_name
    docker rm $container_name
done

echo "All containers have been stopped and removed."


# Sample cmd:
#   ./server.sh --model-repo /home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-8B_bfloat16_2GPU --tp 2 --instance 4
