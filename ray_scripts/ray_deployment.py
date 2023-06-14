from typing import Dict, List

import torch
from ray import serve
from starlette.requests import Request
from transformers import AutoModelForCausalLM, AutoTokenizer


@serve.deployment()
class PredictDeployment:
    def __init__(self):
        self.device = "cuda"

    def reconfigure(self, config):
        # self.bloom = BLOOMSharded(config["model_path"])
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"],
            padding_side="left",
            truncation_side="left",
            use_fast=True,
        )
        self.tokenizer.pad_token = self.tokenizer.bos_token

    @serve.batch(max_batch_size=10, batch_wait_timeout_s=0.5)
    async def batch_generate(
        self,
        text_inputs: List[str],
        temperature: float = 1.0,
        max_new_tokens: int = 128,
    ) -> Dict:
        inputs = self.tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        generated_outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        outputs = self.tokenizer.batch_decode(
            generated_outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        torch.cuda.empty_cache()
        return [{"generated_text": output} for output in outputs]

    async def __call__(self, http_request: Request) -> str:
        json_request: str = await http_request.json()

        return await self.batch_generate(json_request["inputs"])


deployment = PredictDeployment.bind()
