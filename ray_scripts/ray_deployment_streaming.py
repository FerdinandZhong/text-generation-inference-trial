import gc
import json
import time
from typing import Dict

import requests
import torch
from ray import serve
from starlette.requests import Request
from starlette.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@serve.deployment(route_prefix="/streaming")
class PredictDeployment:
    def __init__(self, model_id):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left", truncation_side="left", use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.device = "cpu"

    @torch.inference_mode()
    def naive_generate_tokens(self, request, max_length=1024) -> Dict:
        prompt = request["prompt"]
        max_new_tokens = int(request.get("max_new_tokens", 256))
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        for _ in range(max_new_tokens):
            outputs = self.model(input_ids, attention_mask=attention_mask)

            next_id = outputs.logits.argmax(dim=-1)

            next_token = self.tokenizer.batch_decode(next_id[0:])[-1]

            yield next_token

            if next_token == self.tokenizer.eos_token:
                break
            else:
                input_ids = torch.cat([input_ids, next_id[:, -1].unsqueeze(0)], 1)
                attention_mask = torch.cat([attention_mask, torch.ones([1, 1])], 1)
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_stream(self, request, context_length=2048, stream_interval=1):
        prompt = request["prompt"]
        temperature = float(request.get("temperature", 1.0))
        repetition_penalty = float(request.get("repetition_penalty", 1.0))
        top_p = float(request.get("top_p", 1.0))
        top_k = int(request.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(request.get("max_new_tokens", 256))
        stop_token_ids = request.get("stop_token_ids", None) or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )

        if self.model.config.is_encoder_decoder:
            max_src_len = context_length
        else:
            max_src_len = context_length - max_new_tokens - 8

        input_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_src_len,
        ).input_ids
        input_echo_len = len(input_ids)
        output_ids = list(input_ids)

        if self.model.config.is_encoder_decoder:
            encoder_output = self.model.encoder(
                input_ids=torch.as_tensor([input_ids], device=self.device)
            )[0]
            start_ids = torch.as_tensor(
                [[self.model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=self.device,
            )

        past_key_values = out = None
        for i in range(max_new_tokens):
            if i == 0:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=start_ids,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(
                        input_ids=torch.as_tensor([input_ids], device=self.device),
                        use_cache=True,
                    )
                    logits = out.logits
                past_key_values = out.past_key_values
            else:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=torch.as_tensor([[token]], device=self.device),
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )

                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(
                        input_ids=torch.as_tensor([[token]], device=self.device),
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = out.logits
                past_key_values = out.past_key_values

            if logits_processor:
                if repetition_penalty > 1.0:
                    tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[
                    0
                ]
            else:
                last_token_logits = logits[0, -1, :]

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0:
                tmp_output_ids = output_ids[input_echo_len:]

                output = self.tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )

                result = {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }
                yield json.dumps(result).encode()

            if stopped:
                break

        # finish stream event, which contains finish reason
        if i == max_new_tokens - 1:
            finish_reason = "length"
        elif stopped:
            finish_reason = "stop"
        else:
            finish_reason = None

        result = {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }
        yield json.dumps(result).encode()

        # clean
        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()

    async def __call__(self, http_request: Request) -> str:
        json_request: str = await http_request.json()

        gen = self.generate_stream(json_request)
        return StreamingResponse(gen, status_code=200, media_type="text/plain")


serve.run(PredictDeployment.bind(model_id="bigscience/bloom-560m"))

r = requests.post(
    "http://localhost:8000/streaming",
    json={"prompt": "Tell me a story in english: ", "max_new_tokens": 64},
    stream=True,
)
start = time.time()
r.raise_for_status()
for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
    print(f"Got result {round(time.time()-start, 1)}s after start: {repr(chunk)}")
# __end_example__
