export OMINI_CONFIG=/home/sjtseng0924/OminiControl_test/train/config/config.yaml
export CUDA_VISIBLE_DEVICES=2
# *[Specify the WANDB API key]
export WANDB_API_KEY='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'

echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

# Ensure the repo is on PYTHONPATH so `import omini` works
export PYTHONPATH=/home/sjtseng0924/OminiControl_test:${PYTHONPATH}
cd /home/sjtseng0924/OminiControl_test
# For single-GPU or multi-GPU with accelerate; switch module to train_custom
accelerate launch --num_processes=1 --mixed_precision=bf16 --main_process_port 41358 -m omini.train_flux.train_custom