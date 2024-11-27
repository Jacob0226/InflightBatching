#!/bin/bash

# Sample cmd:
# ./main.sh  --server vLLM
# ./main.sh  --server Triton (Not supported)


function show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help                  Show this help message and exit"
    echo "  --server ServerType         Choices=[vLLM, Triton]. Note: Triton is Triton-Inference-Server"
    echo "  --fp16|--fp8                Run Llama3.1 fp16 or fp8 models."
    echo "E.g., "
    echo "  ./benchmark_scan.sh --server vLLM --fp16"
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
        --fp16)
            Benchmark_Target=("meta-llama/Llama-3.1-8B 1" "meta-llama/Llama-3.1-8B 2" "meta-llama/Llama-3.1-70B 4")
            shift
            ;;
        --fp8)
            Benchmark_Target=("amd/Llama-3.1-8B-Instruct-FP8-KV 1" "amd/Llama-3.1-8B-Instruct-FP8-KV 2" 
                  "amd/Llama-3.1-70B-Instruct-FP8-KV 4")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$server" || -z "$Benchmark_Target" ]]; then
    echo "Error: Both --server and --fp16|--fp8 are required."
    show_help
fi

if [ "${server}" == "vLLM" ]; then
    endpoint=/v1/completions
elif [ "${server}" == "Triton" ]; then
    endpoint=/v2/models/ensemble/generate_stream
fi


for target in "${Benchmark_Target[@]}"; do
    read model_name tp <<< "$target"
    bench_cmd="./benchmark.sh --server ${server} --model ${model_name} --tp ${tp} &"
    echo "bench_cmd=${bench_cmd}"
    eval "${bench_cmd}"
done

