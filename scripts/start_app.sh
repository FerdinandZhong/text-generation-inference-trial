# !/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=/root/text-generation-inference-trial

# update nginx configuration
conda activate server-app
python /root/text-generation-inference-trial/scripts/nginx_config.py --conda_env server-env
# start nginx server
nginx &

# starts ray dashboard
ray start --head --dashboard-port 8265 --num-gpus 2 --disable-usage-stats;

# starts model
conda activate text-generation-env
text-generation-launcher --model-id /root/model_assets/bloomz-7b1 --num-shard 2 --port 5000