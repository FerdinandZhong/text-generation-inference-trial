from typing import Dict, List

import torch
from ray import serve
from starlette.requests import Request
from starlette.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import time


@serve.deployment(route_prefix="/streaming")
class PredictDeployment:
    def __init__(self, model_id):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            truncation_side="left",
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.model.eval()

    def generate_tokens(
        self, text_inputs: List[str], max_token_length: int = 16
    ) -> Dict:
        inputs = self.tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        for _ in range(max_token_length):
            outputs = self.model(
                input_ids, attention_mask=attention_mask
            )

            next_id = outputs.logits.argmax(dim=-1)

            next_token = self.tokenizer.batch_decode(next_id[0:])[-1]

            yield next_token

            if next_token == self.tokenizer.eos_token:
                break
            else:
                input_ids = torch.cat([input_ids, next_id[:, -1].unsqueeze(0)], 1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones([1, 1])], 1
                )
        torch.cuda.empty_cache()

    async def __call__(self, http_request: Request) -> str:
        json_request: str = await http_request.json()

        gen = self.generate_tokens([json_request["inputs"]])
        return StreamingResponse(gen, status_code=200, media_type="text/plain")


serve.run(PredictDeployment.bind(model_id="bigscience/bloom-560m"))

r = requests.post("http://localhost:8000/streaming", json={"inputs": "Hello"}, stream=True)
start = time.time()
r.raise_for_status()
for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
    print(f"Got result {round(time.time()-start, 1)}s after start: {repr(chunk)}")
# __end_example__
