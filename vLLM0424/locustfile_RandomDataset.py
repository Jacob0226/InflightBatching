from locust import HttpUser, task, events, constant
from locust.runners import MasterRunner, WorkerRunner
from openai import OpenAI
import os
import json
import orjson
import time
import threading
from transformers import AutoTokenizer
import random
from datetime import datetime
import pytz
import logging
import requests
import numpy as np
import uuid

worker_data = {"#Req": [], "E2E": [], "TTFT": [], "TPOT": [], "Target":None}
master_data = {"#Req": [], "E2E": [], "TTFT": [], "TPOT": [], "Target":None}
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

class LLMUser(HttpUser):
    # RPS / QPS
    # wait_time = constant(10)  # After the task is finished, wait n second
    # wait_time = constant_pacing(10)  # the task will always be executed every n seconds, no matter the task execution time
    UserIndex = 0  # Static variable to keep track of user index assignment

    def on_start(self):
        self.server = self.environment.parsed_options.server
        self.endpoint = self.environment.parsed_options.endpoint
        self.output_tokens = self.environment.parsed_options.olen
        self.target = self.environment.parsed_options.target
        self.n_completed_request = 0
        # self.id = LLMUser.UserIndex
        # LLMUser.UserIndex += 1
        self.id = uuid.uuid4().int & ((1 << 32) - 1) # Only get the first 32 bits as the user id
        np.random.seed(self.id)  # Set the random seed here

        # Load dataset
        self.input_datafile = self.environment.parsed_options.ifile    
        ilen = int(self.environment.parsed_options.ifile.split('/')[-1].split('.')[0]) # a/b/c/d/2000.txt

        # Load the tokenizer
        self.n_text=200
        try:
            print(self.environment.parsed_options.hf_model, flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.environment.parsed_options.hf_model)
            vocab_size = self.tokenizer.vocab_size
            token_ids_batch = np.random.randint(0, vocab_size, (self.n_text, ilen), dtype=np.int32) # [100, len]
            self.text = []
            for token_ids in token_ids_batch:
                self.text.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
            # print(f"[DEBUG] user id = {self.id}, text={self.text}")

        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print(f"Please ensure the model path is correct and the model is a valid Hugging Face model.")
            exit(1)

               
        server_common_config = {
            "max_tokens": self.output_tokens, 
            "stream": True,                   
            "n": 1,   # Number of generated sequences per request. (Default=1)                        
            "temperature": 0,                 
        }
        vLLM_config = {
            "model": self.environment.parsed_options.hf_model,
            "messages": [
                {"role": "user", "content": self.text[self.n_completed_request]}
            ],
            "ignore_eos": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True
            }                
        }
        
        if self.server == "vLLM":
            self.data = server_common_config | vLLM_config

        self.E2E = []
        self.TTFT = []
        self.TPOT = []

        
        # self.id = LLMUser.UserIndex
        # LLMUser.UserIndex += 1
        # print(f"[DEBUG] user id = {self.id}")

    def parse_output_json(self, data):
        # print(f"[DEBUG] data = {data}, data.choices[0] = {data['choices'][0]}")
        # print(f"[DEBUG2] delta = {data['choices'][0]['delta']}")
        # print(f"[DEBUG3] content = {data['choices'][0]['delta']['content']}")
        # E.g., "choices":[{"index":0,"delta":{"content":" won"},"logprobs":null,"finish_reason":null}]
        if len(data['choices'])>0:
            text = data['choices'][0]['delta']['content']
            return text
        return ""
        

    @task
    def SendRequest(self):
        # if self.server == "vLLM":
        #     self.data["prompt"] = self.prompts[self.prompt_idx]
        # elif self.server == "Triton":
        #     self.data["text_input"] = self.prompts[self.prompt_idx]
        # self.prompt_idx=(1+self.prompt_idx)%100
        
        headers = {"Content-Type": "application/json"}
        self.data["messages"][0]["content"] = self.text[(self.n_completed_request)%self.n_text]
        # print(f"[DEBUG] user id = {self.id}, text={self.data["messages"][0]["content"]}, "
        #       f"n_completed_request={self.n_completed_request}")
        with self.client.post(
            self.endpoint,
            data=json.dumps(self.data),
            headers=headers,
            stream=True,
            catch_response=True,
        ) as response:

            combined_text = ""
            done = False
            
            try:
                response.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Error in response: {response.text}") from e

            for chunk in response.iter_lines(delimiter=b"\n\n"):
                if len(chunk) == 0:
                    # print("come providers send empty lines between data chunks")
                    continue 
                if done:
                    if chunk != b"data: [DONE]":
                        print(f"WARNING: Received more chunks after [DONE]: {chunk}")

                try:
                    now = time.perf_counter()

                    # Stream
                    assert chunk.startswith(b"data:"), f"Unexpected chunk not starting with 'data': {chunk}"
                    chunk = chunk[len(b"data:") :]
                    if chunk.strip() == b"[DONE]":
                        done = True
                        continue
                    data = orjson.loads(chunk)
                    text = self.parse_output_json(data)
                    combined_text += text

                    # The last token comes with OpenAI metric
                    if 'usage' in data and 'server_ttft' in data['usage']:
                        self.TTFT.append(data['usage']['server_ttft'])
                        self.E2E.append(data['usage']['server_e2e_latency'])
                        tpot = (self.E2E[-1] - self.TTFT[-1]) / (data['usage']['completion_tokens']-1)
                        self.TPOT.append(tpot)
                        self.n_completed_request += 1
                        # print(f"[DEBUG] usage exists, self.TTFT={self.TTFT}, self.E2E={self.E2E}"
                        #       f"self.TPOT={self.TPOT}", flush=True)

                except Exception as e:
                    print(f"Failed to parse response: {chunk} with error {repr(e)}")
                    response.failure(e)
                    return


            # print(f"User #{self.id}, req #{self.n_completed_request}, "
            #       f"E2E_Latency(s) = {E2E_Latency:.2f}, TTFT(s)={TTFT:.2f}, TPOT(s)={TPOT:.3f} \n"
            #       f"# output tokens: {len(self.tokenizer.encode(combined_text))}, required output tokens: {self.output_tokens} \n"
            #       f"Prompt: {self.data['prompt'][:1000]}  \n"
            #       f"Generated tokens: {combined_text}\n"
            #       f"------------------\n")


    def on_stop(self):
        # Summarize the benchmark
        if self.n_completed_request == 0:
            print(f"Warning: User {self.id}, #completed_request is 0. Increase [-t] or reduce user number.")
            avg_E2E = avg_TTFT = avg_TPOT = 0
        else:
            avg_E2E = sum(self.E2E) / self.n_completed_request
            avg_TTFT = sum(self.TTFT) / self.n_completed_request
            avg_TPOT = sum(self.TPOT) / self.n_completed_request
        
        worker_data["#Req"].append(self.n_completed_request)
        worker_data["E2E"].append(avg_E2E)
        worker_data["TTFT"].append(avg_TTFT)
        worker_data["TPOT"].append(avg_TPOT)
        worker_data["Target"] = self.target


def report_metrics_to_master(environment, msg):
    global master_data
    master_data["#Req"]+=msg.data["#Req"]
    master_data["E2E"]+=msg.data["E2E"]
    master_data["TTFT"]+=msg.data["TTFT"]
    master_data["TPOT"]+=msg.data["TPOT"]
    master_data["Target"]=msg.data["Target"]
    # print(f"[DEBUG] msg={msg.data}")
    # print(f"  master_data={master_data}")
    
    n_user = len(master_data["#Req"])
    if n_user == 0: # When user = 1, processes = 4, not every process handle at least 1 user.
        return
    n_req = sum(master_data["#Req"])
    # print("Only show the first 8 users results:")
    # for i in range(min(n_user,8)):
    #     print(f"User #{i:3d}, Completed Request = {master_data['#Req'][i]}, "
    #         f"E2E(s) = {master_data['E2E'][i]:.2f}, "
    #         f"TTFT(s) = {master_data['TTFT'][i]:.2f}, "
    #         f"TPOT(s) = {master_data['TPOT'][i]:.3f} ")

    ave_E2E = sum(master_data['E2E']) / n_user
    ave_TTFT = sum(master_data['TTFT']) / n_user
    ave_TPOT = sum(master_data['TPOT']) / n_user

    # Save result
    # Save to a JSON file
    if os.path.exists(environment.parsed_options.outJson):
        with open(environment.parsed_options.outJson, 'r') as f:
            data = json.load(f)  # Load JSON data as a Python dictionary
    else:
        data = dict()
        data["vLLM"] = dict()
    server = environment.parsed_options.server
    key = environment.parsed_options.target
    model_name = os.path.dirname(key)
    test_case = os.path.basename(key)
    if server not in data:
        data[server] = {}
    if model_name not in data[server]:
        data[server][model_name] = {}
    
    gmt_plus_8 = pytz.timezone('Etc/GMT-8')
    current_time = datetime.now(gmt_plus_8)
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M')
    date = "Taipei Time: " + formatted_time

    data[server][model_name][test_case] = {
        "Date": date,
        "#Req": n_req,
        "E2E": ave_E2E,
        "TTFT": ave_TTFT,
        "TPOT": ave_TPOT
    }
    # print(f"data = {data}")

    dir = os.path.dirname(environment.parsed_options.outJson)
    os.makedirs(dir, exist_ok=True)  
    with open(environment.parsed_options.outJson, 'w') as f:
        json.dump(data, f, indent=4)  # Save with indentation for readability
        # print(f"[DEBUG] json data=", data)
        # print(f"[DEBUG] outJson={outJson}")

@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    if isinstance(environment.runner, MasterRunner):
        environment.runner.register_message('report_metrics_to_master', report_metrics_to_master)
        if environment.parsed_options.rpd_profile:
            logger.info(f"host = {environment.parsed_options.host} ")
            response = requests.post(environment.parsed_options.host+"/start_profile")
            if response.status_code == 200:
                logger.info(f"Start rpd profiling ")
            else:
                logger.info(f"Fail to start rpd profiling ")
    elif isinstance(environment.runner, WorkerRunner):
        pass

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    if isinstance(environment.runner, MasterRunner):
        #print("[DEBUG] I'm on master node")
        if environment.parsed_options.rpd_profile:
            response = requests.post(environment.parsed_options.host+"/stop_profile")
            if response.status_code == 200:
                logger.info(f"Stop rpd profiling ")
            else:
                logger.info(f"Fail to end rpd profiling ")
    elif isinstance(environment.runner, WorkerRunner):
        # print(f"[DEBUG] I'm on worker node.")
        environment.runner.send_message('report_metrics_to_master', worker_data)


@events.init_command_line_parser.add_listener
def init_parser(parser):  
    parser.add_argument(
        "--server",
        type=str,
        choices=['vLLM'],
        required=True,
        help="Choices=[vLLM]. Only support vLLM now",
    ) 
    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        help="The endpoint on server, e.g., /v1/chat/completions",
    ) 
    parser.add_argument(
        "-ifile",
        type=str,
        required=True,
        help="file of input txt",
    )
    parser.add_argument(
        "-olen",
        type=int,
        required=True,
        help="file of input txt",
    )
    parser.add_argument(
        "-outJson",
        type=str,
        default="benchmark.json",
        help="file of output json to store metrics",
    )
    parser.add_argument(
        "-target",
        type=str,
        help="saving result in the specified key 'target' in outJson file",
    )
    parser.add_argument(
        "-m",
        "--hf-model",
        env_var="MODEL_PATH",
        type=str,
        help=("Path to HF model folder. If not specified we will use env var MODEL_PATH.\n"
              "Required by vLLM server. Also, we use tokenizer to debug the token lengths.")
    )
    parser.add_argument("--rpd-profile", action="store_true", help="Activate profiling for ROCm/vLLM")

