from locust import HttpUser, task, events, constant
from locust.runners import MasterRunner, WorkerRunner
from openai import OpenAI
import os
import json
import orjson
import time
import threading
from transformers import AutoTokenizer

class LLMUser(HttpUser):
    # RPS / QPS
    # wait_time = constant(n)  # After the task is finished, wait n second
    # wait_time = constant_pacing(10)  # the task will always be executed every n seconds, no matter the task execution time
    UserIndex = 0  # Static variable to keep track of user index assignment

    def on_start(self):
        self.server = self.environment.parsed_options.server
        self.endpoint = self.environment.parsed_options.endpoint
        self.output_tokens = self.environment.parsed_options.olen
        self.output_folder = self.environment.parsed_options.out
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)

        # Load dataset
        self.input_datafile = self.environment.parsed_options.ifile      
        with open(self.input_datafile, 'r') as file:
            self.prompt = file.read() 
               
        server_common_config = {
            "max_tokens": self.output_tokens, 
            "stream": True,                   
            "n": 1,   # Number of generated sequences per request. (Default=1)                        
            "temperature": 0,                 
        }
        vLLM_config = {
            "model": self.environment.parsed_options.hf_model,
            "prompt": self.prompt,      # "What is machine learning?"      
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
            #       f"Generated tokens: {combined_text}")
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
        
        if self.id<8:
            print("Only show the first 8 users results:")
            print(f"User #{self.id:3d}, Completed Request = {self.n_completed_request:3d}, "
                f"avg_E2E_Latency(s) = {avg_E2E:.2f}, avg_TTFT(s) = {avg_TTFT:.2f}, avg_TPOT(s) = {avg_TPOT:.3f} ")

        # Save result
        # Save to a JSON file
        if self.output_folder:
            json_file = os.path.join(self.output_folder,  f"benchmark_{self.id:02}.json")
            with open(json_file, 'w') as json_file:
                data = {
                    "E2E": self.E2E,
                    "TTFT": self.TTFT,
                    "TPOT": self.TPOT
                }
                json.dump(data, json_file, indent=4)  # 'indent=4' for pretty formatting



@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("[DEBUG] A new test is starting")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    if isinstance(environment.runner, MasterRunner):
        print("[DEBUG] I'm on master node")
    elif isinstance(environment.runner, WorkerRunner):
        print("[DEBUG] I'm on worker node")
    # if not isinstance(environment.runner, MasterRunner):
    #     print("A new test is starting")
    #     environment.runner.register_message('test_users', setup_test_users)
    # if not isinstance(environment.runner, WorkerRunner):
    #     environment.runner.register_message('acknowledge_users', on_acknowledge)



@events.init_command_line_parser.add_listener
def init_parser(parser):  
    parser.add_argument(
        "--server",
        type=str,
        choices=['vLLM', 'Triton'],
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
        help="file of input txt",
    )
    parser.add_argument(
        "-olen",
        type=int,
        help="file of input txt",
    )
    parser.add_argument(
        "-out",
        type=str,
        help="saving result in the output folder",
    )
    parser.add_argument(
        "-m",
        "--hf-model",
        env_var="MODEL_PATH",
        type=str,
        help=("Path to HF model folder. If not specified we will use env var MODEL_PATH.\n"
              "Required by vLLM server. Also, we use tokenizer to debug the token lengths.")
    )

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
