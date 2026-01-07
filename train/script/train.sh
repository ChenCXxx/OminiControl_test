export OMINI_CONFIG=/home/chchen/lab/OminiControl_test/train/config/config.yaml
export CUDA_VISIBLE_DEVICES=5
# *[Specify the WANDB API key]
export WANDB_API_KEY='your_wandb_api_key_here'

echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

# Ensure the repo is on PYTHONPATH so `import omini` works
export PYTHONPATH=/home/chchen/lab/OminiControl_test:${PYTHONPATH}

cd /home/chchen/lab/OminiControl_test
# For single-GPU or multi-GPU with accelerate; switch module to train_custom
accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port 41358 -m omini.train_flux.train_custom