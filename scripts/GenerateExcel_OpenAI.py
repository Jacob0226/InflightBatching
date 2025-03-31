import os
import re
import json
import argparse
from datetime import datetime
import pytz



def parse_log_file(filepath, grouped_metrics):
    """Parses a single log file and extracts relevant metrics."""
    generation_tokens = 0.0  # Initialize generation tokens
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("vllm:request_success_total{finished_reason=\"length\""):
                value = float(line.split()[-1])
                grouped_metrics["request_success_total"].append(value)
            elif line.startswith("vllm:prompt_tokens_total"):
                value = float(line.split()[-1])
                grouped_metrics["prompt_tokens_total"].append(value)
            elif line.startswith("vllm:generation_tokens_total"):
                value = float(line.split()[-1])
                grouped_metrics["generation_tokens_total"].append(value)
            elif line.startswith("vllm:iteration_tokens_total_sum"):
                value = float(line.split()[-1])
                grouped_metrics["iteration_tokens_total_sum"].append(value)
            elif line.startswith("vllm:request_inference_time_seconds_sum"):
                value = float(line.split()[-1])
                grouped_metrics["request_inference_time_seconds_sum"].append(value)
            elif line.startswith("vllm:e2e_request_latency_seconds_sum"):
                value = float(line.split()[-1])
                grouped_metrics["e2e_request_latency_seconds_sum"].append(value)
            elif line.startswith("vllm:time_to_first_token_seconds_sum"): 
                value = float(line.split()[-1])
                grouped_metrics["time_to_first_token_seconds_sum"].append(value)

def process_log_files(log_folder, n_server):
    model_name = log_folder.split('/')[1]
    final_metrics = {model_name: []}

    # filenames: 10s_i2000_01user_server0.log
    TIME = "3m"
    N_USER = [1, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 196]
    IO_LEN = ["i2000", "i4400", "i8600"]

    for n_user in N_USER:
        for io_len in IO_LEN:
            grouped_metrics = {
                "request_success_total": [],
                "prompt_tokens_total": [],
                "generation_tokens_total": [],
                "iteration_tokens_total_sum": [],
                "request_inference_time_seconds_sum": [],
                "e2e_request_latency_seconds_sum": [],
                "time_to_first_token_seconds_sum": [],
            }

            exist = True
            for server_idx in range(n_server):
                log_filename = f"{TIME}_{io_len}_{n_user:02d}user_server{server_idx}.log"
                log_path = os.path.join(log_folder, log_filename)

                if os.path.exists(log_path) == False:
                    exist = False
                    # print(f"{log_path} not existed, skip...")
                else:
                    print(log_path)
                    parse_log_file(log_path, grouped_metrics)
            
            if exist:
                grouped_metrics["request_success_total"] = sum(grouped_metrics["request_success_total"])
                grouped_metrics["prompt_tokens_total"] = sum(grouped_metrics["prompt_tokens_total"])
                grouped_metrics["generation_tokens_total"] = sum(grouped_metrics["generation_tokens_total"])
                grouped_metrics["iteration_tokens_total_sum"] = sum(grouped_metrics["iteration_tokens_total_sum"])
                grouped_metrics["request_inference_time_seconds_sum"] = sum(grouped_metrics["request_inference_time_seconds_sum"]) / grouped_metrics["request_success_total"]
                grouped_metrics["e2e_request_latency_seconds_sum"] = sum(grouped_metrics["e2e_request_latency_seconds_sum"]) / grouped_metrics["request_success_total"]
                grouped_metrics["time_to_first_token_seconds_sum"] = sum(grouped_metrics["time_to_first_token_seconds_sum"]) / grouped_metrics["request_success_total"]

                final_metrics[model_name].append({f"{TIME}_{io_len}_{n_user:02d}user": grouped_metrics})
    return final_metrics

def main():
    parser = argparse.ArgumentParser(description="Process log files and generate a JSON report.")
    parser.add_argument("--folder", required=True, help="The root folder containing the log files (e.g., MultiServer_vLLM/...).")
    args = parser.parse_args()

    input_folder = args.folder
    inference_type = ""

    if "MultiServer_vLLM" in input_folder:
        inference_type = "vLLM"
    elif "MultiServer_Triton" in input_folder:
        inference_type = "Triton"
    else:
        print("Warning: Could not determine inference type from folder name. Setting to UNKNOWN.")
        inference_type = "UNKNOWN"

    openai_metric_folder = os.path.join(input_folder, "OpenAI_Metric")
    os.makedirs(openai_metric_folder, exist_ok=True)
    output_filepath = os.path.join(input_folder, "OpenAI_Metric.json")

    num_servers=1
    match = re.search(r'(\d+)xTP', input_folder)
    if match:
        num_servers = int(match.group(1))
    else:
        print("Warning: Could not determine the number of servers from the folder name (e.g., '4xTP1').")
    print("Num of server =",num_servers)

    aggregated_data = process_log_files(openai_metric_folder, num_servers)

    output_json = {"Inference": inference_type, "Data": aggregated_data}

    with open(output_filepath, 'w') as outfile:
        json.dump(output_json, outfile, indent=4)
        # print(output_json)

    print(f"Aggregated {inference_type} metrics saved to: {output_filepath}")

if __name__ == "__main__":
    main()

'''

python GenerateExcel_OpenAI.py --folder MultiServer_vLLM/meta-llama_Llama-3.1-8B_4xTP1
python GenerateExcel_OpenAI.py --folder MultiServer_vLLM/0331_meta-llama_Llama-3.1-8B_4xTP1



'''