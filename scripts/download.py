# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

from requests.exceptions import HTTPError


def hf_download() -> None:
    from huggingface_hub import snapshot_download

    MODEL_PATH = os.getenv('MODEL_PATH')
    MODEL_ID = os.getenv('MODEL_ID')
    if MODEL_PATH==None:
        print("env variable MODEL_PATH is None. Export MODEL_PATH=/Path/to/save/model")
        exit(0)
    if MODEL_ID==None:
        print("env variable MODEL_ID is None. Export MODEL_ID=01-ai/Yi-34B")
        exit(0)
    os.makedirs(f"{MODEL_PATH}", exist_ok=True)

    try:
        snapshot_download(MODEL_ID, 
            local_dir=f"{MODEL_PATH}",
            local_dir_use_symlinks=False,
            #ignore_patterns="*.safetensors"
            )
    except HTTPError as e:
        if e.response.status_code == 401:
            print("You need to export a valid token 'export HF_TOKEN=XXXXX' to download private checkpoints.")
        else:
            raise e

if __name__ == '__main__':
    hf_download()

