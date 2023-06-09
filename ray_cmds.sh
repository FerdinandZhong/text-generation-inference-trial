#!/bin/bash

export MASTER_ADDR=localhost   
export MASTER_PORT=29500 
export RANK=2
export SAFETENSORS_FAST_GPU=1
export WORLD_SIZE=2
ray start --head --dashboard-port 8265 --num-gpus 1 --disable-usage-stats;
serve deploy /root/text-generation-inference-trial/tests/ray_config.yaml -a http://127.0.0.1:52365