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

worker_data = {"#Req": [], "E2E": [], "TTFT": [], "TPOT": [], "Target":None}
master_data = {"#Req": [], "E2E": [], "TTFT": [], "TPOT": [], "Target":None}
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

class LLMUser(HttpUser):
    # RPS / QPS
    # wait_time = constant(n)  # After the task is finished, wait n second
    # wait_time = constant_pacing(10)  # the task will always be executed every n seconds, no matter the task execution time
    UserIndex = 0  # Static variable to keep track of user index assignment

    def on_start(self):
        self.server = self.environment.parsed_options.server
        self.endpoint = self.environment.parsed_options.endpoint
        self.output_tokens = self.environment.parsed_options.olen
        self.target = self.environment.parsed_options.target

        # Load dataset
        self.input_datafile = self.environment.parsed_options.ifile      
        with open(self.input_datafile, 'r') as file:
            self.prompt = file.read() 
        self.words = self.prompt.split()
        self.prompts = []
        for i in range(100): # Generate random prompt
            random.shuffle(self.words)
            self.prompts.append(' '.join(self.words))
        self.prompt_idx = 0

               
        server_common_config = {
            "max_tokens": self.output_tokens, 
            "stream": True,                   
            "n": 1,   # Number of generated sequences per request. (Default=1)                        
            "temperature": 0,                 
        }
        vLLM_config = {
            "model": self.environment.parsed_options.hf_model,
            "prompt": self.prompt,
            "ignore_eos": True                
        }
        triton_config = {
            "text_input": self.prompt,        
            "min_length": self.output_tokens, 
        }
        
        if self.server == "vLLM":
            self.data = server_common_config | vLLM_config
        elif self.server == "Triton":
            self.data = server_common_config | triton_config

        if self.environment.parsed_options.hf_model is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.environment.parsed_options.hf_model)

        # Result:
        self.E2E = []
        self.TTFT = []
        self.TPOT = []

        self.id = LLMUser.UserIndex
        self.n_completed_request = 0
        LLMUser.UserIndex += 1

    def parse_output_json(self, data):
        # Triton inference server
        if self.server == 'Triton':
            # e.g., {"model_name":"ensemble","model_version":"1","sequence_end":false,"sequence_id":0,"sequence_index":0,"sequence_start":false,"text_output":" What"}' with error KeyError('choices'
            return data["text_output"]
        elif self.server == 'vLLM':
            assert len(data["choices"]) == 1, f"Too many choices {len(data['choices'])}"
            choice = data["choices"][0]
            text = choice["text"]
            return text
        

    @task
    def SendRequest(self):
        # if self.server == "vLLM":
        #     self.data["prompt"] = self.prompts[self.prompt_idx]
        # elif self.server == "Triton":
        #     self.data["text_input"] = self.prompts[self.prompt_idx]
        # self.prompt_idx=(1+self.prompt_idx)%100
        
        t_start = time.perf_counter()
        with self.client.post(
            self.endpoint,
            data=json.dumps(self.data),
            stream=True,
            catch_response=True,
        ) as response:

            combined_text = ""
            t_first_token = None
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

                    if combined_text and t_first_token is None:
                        t_first_token = now

                except Exception as e:
                    print(f"Failed to parse response: {chunk} with error {repr(e)}")
                    response.failure(e)
                    return

            now = time.perf_counter()
            E2E_Latency = now - t_start
            TTFT = t_first_token - t_start
            GenerationT = now - t_first_token
            TPOT = GenerationT / (self.output_tokens-1)
            # print(f"User #{self.id}, req #{self.n_completed_request}, "
            #       f"E2E_Latency(s) = {E2E_Latency:.2f}, TTFT(s)={TTFT:.2f}, TPOT(s)={TPOT:.3f} \n"
            #       f"# output tokens: {len(self.tokenizer.encode(combined_text))}, required output tokens: {self.output_tokens} \n"
            #       f"Prompt: {self.data['prompt'][:1000]}  \n"
            #       f"Generated tokens: {combined_text}\n"
            #       f"------------------\n")
            self.n_completed_request += 1
            self.E2E.append(E2E_Latency)
            self.TTFT.append(TTFT)
            self.TPOT.append(TPOT)
            # print("=======================================================")

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
        data["Triton"] = dict()
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

    with open(environment.parsed_options.outJson, 'w') as f:
        json.dump(data, f, indent=4)  # Save with indentation for readability

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
        choices=['vLLM', 'Triton'],
        required=True,
        help="Choices=[vLLM, Triton]. Note: Triton is Triton-Inference-Server",
    ) 
    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        help="The endpoint on server, e.g., /v1/completions",
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

'''
vLLM:
locust --server vLLM --hf-model $MODEL_PATH \
    --host http://localhost:${SERVER_PORT} --endpoint /v1/completions \
    -t 10s -u 2 -r 2 \
    -ifile Datasets/2500.txt -olen 350 -out report/ \
    --headless --only-summary --csv report/report --html report/report.html


Triton-Inference-Server:
locust --server Triton --hf-model $MODEL_PATH \
    --host http://localhost:${SERVER_PORT} --endpoint /v2/models/ensemble/generate_stream \
    -t 10s -u 2 -r 2 \
    -ifile Datasets/2500.txt -olen 350 -out report/ \
    --headless --only-summary --csv report/report --html report/report.html

'''
