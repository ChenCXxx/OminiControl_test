import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
import time

from typing import List

import prodigyopt

from ..pipeline.flux_omini import transformer_forward, encode_images

from transformers import AutoTokenizer, LongT5ForConditionalGeneration

def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = ["default", None],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.device_torch = torch.device(device)
        self.dtype_torch = dtype

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype
        ).to(device)

        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Load LongT5 model for text encoding
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"
        # )
        # self.longt5 = LongT5ForConditionalGeneration.from_pretrained(
        #     "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"
        # ).to(device=device, dtype=dtype).eval()

        # for p in self.longt5.parameters():  # Freeze LongT5 parameters
        #     p.requires_grad_(False)

        # Projection layers
        # t5_hidden = self.longt5.config.d_model  # typically 1024
        # flux_token_dim = self.transformer.context_embedder.in_features  # flux: 4096
        # self.t5_proj = torch.nn.Linear(
        #     t5_hidden, flux_token_dim, bias=False
        # ).to(device=device, dtype=dtype)

        # self.t5_pool_proj = torch.nn.Linear(
        #     flux_token_dim, 768, bias=False
        # ).to(device=device, dtype=dtype)
        # self.t5_proj_params = list(self.t5_proj.parameters()) + list(self.t5_pool_proj.parameters())


        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None])
        self.active_adapter = next((a for a in self.adapter_names if a is not None), "default")
        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)


    def _resolve_local_lora(self, path_or_dir: str) -> tuple[str, str]:
        """
        回傳 (dir, weight_name)。支援：
          - 直接給 .safetensors 檔案
          - 給資料夾（優先找 default.safetensors，否則取第一個 .safetensors）
        若不是本地路徑，回傳 (None, None)（代表可能是 HF repo id）
        """
        if os.path.isfile(path_or_dir) and path_or_dir.endswith(".safetensors"):
            return os.path.dirname(path_or_dir), os.path.basename(path_or_dir)
        if os.path.isdir(path_or_dir):
            cands = [n for n in os.listdir(path_or_dir) if n.endswith(".safetensors")]
            weight = "default.safetensors" if "default.safetensors" in cands \
                     else (cands[0] if cands else None)
            if weight is None:
                raise FileNotFoundError(f"No .safetensors found under: {path_or_dir}")
            return path_or_dir, weight
        return None, None
    

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        adapter_name = self.active_adapter
        target_regex = (lora_config or {}).get("target_modules", ".*")

        if lora_path:
            # TODO: Implement this
            local_dir, weight_name = self._resolve_local_lora(lora_path)
            if local_dir is not None:
                self.flux_pipe.load_lora_weights(local_dir, weight_name=weight_name, adapter_name=adapter_name)
                print(f"[LoRA] Loaded from local: {os.path.join(local_dir, weight_name)} "
                      f"as adapter='{adapter_name}'")
                # safep = os.path.join(local_dir, "k_mlp.safetensors")

                # state = safetensors_load(safep)
                # self.k_encoder.load_state_dict(state, strict=True)
                # print(f"[K-MLP] Loaded from {safep}")

            else:
                self.flux_pipe.load_lora_weights_from_hf(
                    lora_path, adapter_name=adapter_name
                )
                print(f"[LoRA] Loaded from HF repo: {lora_path} as adapter='{adapter_name}'")
            # Collect trainable LoRA parameters after loading
            
            lora_layers = self._collect_lora_params_by_config(adapter_name, target_regex)

        else:
            # config: target_modules, r, init_lora_weight...
            # add adapters: insert new lora layers into the transformer
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )
        k_path_st = os.path.join(path, "k_mlp.safetensors")

        state = self.k_encoder.state_dict()

        safetensors_save(state, k_path_st)
        print(f"[K-MLP] Saved to {k_path_st}")

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config
        # param_cfg = opt_config.get("params", {k: v for k, v in opt_config.items() if k != "type"})

        # Set the trainable parameters
        self.trainable_params = list(self.lora_layers)
        # if self.t5_proj_params:
        #     self.trainable_params += list(self.t5_proj_params)

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **param_cfg)
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **param_cfg)
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer

    # def encode_prompt_longt5(self, prompts, max_len=16384):
    #     # encode with longt5
    #     inputs = self.tokenizer(
    #         prompts,
    #         padding="longest",
    #         truncation=True,
    #         max_length=max_len,
    #         return_tensors="pt"
    #     ).to(self.device_torch)

    #     with torch.no_grad():
    #         encoder_t5 = self.longt5.encoder(
    #             input_ids=inputs.input_ids,
    #             attention_mask=inputs.attention_mask,
    #             return_dict=True,
    #         )
        
    #     t5_hidden = encoder_t5.last_hidden_state  # (b, seq, d_model)
    #     token_4096 = self.t5_proj(t5_hidden)  # (b, seq, flux_token_dim)


    #     mask = inputs.attention_mask.unsqueeze(-1)
    #     pooled_4096 = (token_4096 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    #     pooled_768 = self.t5_pool_proj(pooled_4096)  # (b, 768)

    #     seq_len = token_4096.shape[1]
    #     text_ids = torch.zeros((seq_len,3), dtype=torch.long, device=self.device_torch)
    #     text_ids[:, 0] = torch.arange(seq_len, device=self.device_torch)

    #     return token_4096, pooled_768, text_ids

    def training_step(self, batch, batch_idx):
        imgs, prompts = batch["image"], batch["description"]
        image_latent_mask = batch.get("image_latent_mask", None)

        # Get the conditions and position deltas from the batch
        conditions, position_deltas, position_scales, latent_masks = [], [], [], []
        for i in range(1000):
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])
            position_deltas.append(batch.get(f"position_delta_{i}", [[0, 0]]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)

            # Prepare text input
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.flux_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )
            # (
            #     prompt_embeds,
            #     pooled_prompt_embeds,
            #     text_ids,
            # ) = self.encode_prompt_longt5(
            #     prompts,
            #     max_len=self.model_config.get("max_sequence_length", 16384),
            # )


            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device_torch))
            x_1 = torch.randn_like(x_0).to(self.device_torch)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype_torch)
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]
                img_ids = img_ids[image_latent_mask[0]]

            # Prepare conditions
            condition_latents, condition_ids = [], []
            for cond, p_delta, p_scale, latent_mask in zip(
                conditions, position_deltas, position_scales, latent_masks
            ):
                # Prepare conditions
                c_latents, c_ids = encode_images(self.flux_pipe, cond)
                # Scale the position (see OminiConrtol2)
                if p_scale != 1.0:
                    scale_bias = (p_scale - 1.0) / 2
                    c_ids[:, 1:] *= p_scale
                    c_ids[:, 1:] += scale_bias
                # Add position delta (see OminiControl)
                c_ids[:, 1] += p_delta[0][0]
                c_ids[:, 2] += p_delta[0][1]
                if len(p_delta) > 1:
                    print("Warning: only the first position delta is used.")
                # Append to the list
                if latent_mask is not None:
                    c_latents, c_ids = c_latents[latent_mask], c_ids[latent_mask[0]]
                condition_latents.append(c_latents)
                condition_ids.append(c_ids)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device_torch)
                if self.transformer.config.guidance_embeds
                else None
            )

        branch_n = 2 + len(conditions)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device_torch)
        # Disable the attention cross different condition branches
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
        # Disable the attention from condition branches to image branch and text branch
        if self.model_config.get("independent_condition", False):
            group_mask[2:, :2] = False

        # Forward pass
        transformer_out = transformer_forward(
            self.transformer,
            image_features=[x_t, *(condition_latents)],
            text_features=[prompt_embeds],
            img_ids=[img_ids, *(condition_ids)],
            txt_ids=[text_ids],
            # There are three timesteps for the three branches
            # (text, image, and the condition)
            timesteps=[t, t] + [torch.zeros_like(t)] * len(conditions),
            # Same as above
            pooled_projections=[pooled_prompt_embeds] * branch_n,
            guidances=[guidance] * branch_n,
            # The LoRA adapter names of each branch
            adapters=self.adapter_names,
            return_dict=False,
            group_mask=group_mask,
        )
        pred = transformer_out[0]

        # Compute loss
        step_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()

        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}, test_function=None):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.test_function = test_function

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0 and self.test_function:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            pl_module.eval()
            self.test_function(
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
            )
            pl_module.train()


def train(dataset, trainable_model, config, test_function):
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    # config = get_config()

    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_confIIig.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataloader
    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Callbacks for testing and saving checkpoints
    if is_main_process:
        callbacks = [TrainingCallback(run_name, training_config, test_function)]

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks if is_main_process else [],
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)
