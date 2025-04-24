#!/bin/bash
set -x

# vLLM server
# For simplicity on AMD, test 1 instance of the vLLM server 

# 8B-BF16
# ./server.sh --model-name meta-llama/Llama-3.1-8B --v1

# 70B-FP8
# ./server.sh --model-name amd/Llama-3.3-70B-Instruct-FP8-KV # Run with vLLM engine v0
./server.sh --model-name amd/Llama-3.3-70B-Instruct-FP8-KV --v1 # Run with vLLM engine v1
