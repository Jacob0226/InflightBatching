#!/bin/bash
set -x

# vLLM server

# 8B-BF16
./server.sh --model-name meta-llama/Llama-3.1-8B --tp 1 --instance 4
# ./server.sh --model-name meta-llama/Llama-3.1-8B --tp 2 --instance 1

# # 70B-FP8
# ./server.sh --model-name amd/Llama-3.1-70B-Instruct-FP8-KV --tp 4 --instance 1
./server.sh --model-name amd/Llama-3.3-70B-Instruct-FP8-KV --tp 1 --instance 2


# Extra
