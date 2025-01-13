#!/bin/bash
set -x

# Triton server + TRTLLM

# 8B-BF16
# ./server.sh  --instance 4 --tp 2  \
#     --model-repo /home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-8B_bfloat16_2GPU 

# # 70B-FP8
# ./server.sh  --instance 2 --tp 4  \
#     --model-repo /home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-70B_fp8_4GPU

# 70B-FP8 
./server.sh  --instance 1 --tp 8  \
    --model-repo /home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-70B_fp8_8GPU 

# 8B-BF16, 1xTP2
./server.sh  --instance 1 --tp 2  \
    --model-repo /home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-8B_bfloat16_2GPU 

# 8B-BF16, 1xTP1
./server.sh  --instance 1 --tp 1  \
    --model-repo /home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-8B_bfloat16_1GPU 


# Double check
# H100 1xTP4
./server.sh  --tp 4 --instance 1 \
    --model-repo /home/jacchang/POC_RFP/vllm/Llama3.1/Triton_Model_Repo/triton_Llama-3.1-70B_fp8_4GPU


# Test cases
# 8B-BF16
    # MI300X 8xTP1 vs H100 4xTP2
    # MI300 TP1 x2 vs H100 TP2
    # MI300 TP1 vs H100 TP1
# 70B-FP8
    # MI300X 8xTP1 vs H100 2xTP4
        # Good to see this to see how it compares to MI300 TP8 vs H100 TP4 x2
    # MI300 TP2 x2 vs H100 TP4
        # Spoke to Nathan and we added another table to the list
    # MI300 TP1 x4 vs H100 TP4
    # MI300 TP8 vs H100 TP4 x2
    # MI300 TP8 vs H100 TP8
    # MI300X TP4 vs H100 TP4


