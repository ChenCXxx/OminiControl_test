import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
import time
import re
from typing import List

import prodigyopt

from ..pipeline.flux_omini import transformer_forward, encode_images
import math
from typing import List
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

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
# class SiLUBokehKEncoder(torch.nn.Module):
#     def __init__(self, latent_channels=64, mlp_hidden=256, alpha=0.1, dtype=torch.bfloat16):
#         super().__init__()
#         self.alpha = alpha
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(1, mlp_hidden),
#             torch.nn.SiLU(),
#             torch.nn.Linear(mlp_hidden, latent_channels),
#         )
#         # 小心初始化，穩定一點
#         for m in self.mlp.modules():
#             if isinstance(m, torch.nn.Linear):
#                 torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
#                 if m.bias is not None:
#                     torch.nn.init.zeros_(m.bias)

#     def forward(self, K):
#         if K.ndim == 1: K = K[:, None]
#         x = (self.alpha * K).to(torch.bfloat16)   # 相當於 K/=10
#         y = self.mlp(x)
#         return y

# import torch.nn.functional as F
# from torch.nn.utils import parametrize
# import torch.nn as nn
# class SoftplusParam(nn.Module):
#     def forward(self, w):
#         # 將任意實數權重 w_raw 經 softplus 變成非負權重
#         return F.softplus(w)

# class BokehKEncoder(nn.Module):
#     def __init__(self, latent_channels=4096, mlp_hidden=256, alpha=0.1, dtype=torch.bfloat16):
#         super().__init__()
#         self.alpha = alpha
#         self.dtype = dtype

#         self.mlp = nn.Sequential(
#             nn.Linear(1, mlp_hidden),   # 層 0
#             nn.Softplus(),              # 單調激活
#             nn.Linear(mlp_hidden, latent_channels),  # 層 2
#         )

#         # 只改「權重非負」，結構不變
#         parametrize.register_parametrization(self.mlp[0], "weight", SoftplusParam())
#         parametrize.register_parametrization(self.mlp[2], "weight", SoftplusParam())

#         # 初始化：raw 權重靠近 0 更穩（因為最後會 softplus）
#         with torch.no_grad():
#             for m in self.mlp.modules():
#                 if isinstance(m, nn.Linear):
#                     m.weight.zero_()
#                     if m.bias is not None:
#                         m.bias.zero_()

#     def forward(self, K):
#         if K.ndim == 1:
#             K = K[:, None]
        
#         x = (self.alpha * K).to(torch.bfloat16)
#         y = self.mlp(x)
#         return y
# class SiLUBokehKEncoder(torch.nn.Module):
#     """
#     將標量 K -> [sin/cos PE] -> MLP -> latent_channel 維度
#     """
#     def __init__(self, latent_channels: int, pe_dims: int = 64, mlp_hidden: int = 256, dtype=torch.bfloat16):
#         super().__init__()
#         self.latent_channels = latent_channels
#         self.pe_dims = pe_dims
#         self.dtype = dtype

#         in_dim = pe_dims * 2  # sin + cos
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(in_dim, mlp_hidden),
#             torch.nn.SiLU(),
#             torch.nn.Linear(mlp_hidden, latent_channels),
#         )

#         # 小心初始化，穩定一點
#         for m in self.mlp.modules():
#             if isinstance(m, torch.nn.Linear):
#                 torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
#                 if m.bias is not None:
#                     torch.nn.init.zeros_(m.bias)

#     @staticmethod
#     def _posenc(x: torch.Tensor, pe_dims: int):
#         """
#         x: [B] 或 [B,1]
#         回傳: [B, pe_dims*2]，按對數頻率做 sin/cos
#         """
#         if x.ndim == 1:
#             x = x[:, None]
#         device = x.device
#         # 使用 log scale 頻率
#         freqs = torch.exp(torch.linspace(math.log(1.0), math.log(64.0), pe_dims, device=device))
#         phases = x * freqs[None, :]
#         return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)

#     def forward(self, K: torch.Tensor):
#         """
#         K: [B] 或 [B,1]  (float)
#         return: [B, C]
#         """
#         K = K.to(self.dtype)
#         pe = self._posenc(K, self.pe_dims).to(self.dtype)
#         feat = self.mlp(pe).to(self.dtype)    # [B, C]
#         return feat

class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = [None, None, "default"],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype
        ).to(device)
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None])
        self.active_adapter = next(
        (a for a in reversed(adapter_names) if a is not None),  # 生成器：從尾到頭找非 None
        "default"                                               # 找不到時的預設值
        )
        # # ---- 建立 K 編碼器（可訓練）----這裡要調查一下數據!
        # self.k_encoder = BokehKEncoder(
        #     latent_channels=4096, #這裡要變成跟text branch一樣的channel，prompt_embeds.shape=[1,512,4096]
        #     mlp_hidden=self.model_config.get("k_mlp_hidden", 256),
        #     dtype=dtype,
        # ).to(device).to(dtype)
        # self.renderer = ModuleRenderScatter().to(device).eval()
        # self.flux_pipe.k_encoder = self.k_encoder
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

    # ---------- 核心：用你的 target_modules 正則 精準開 requires_grad ----------
    def _collect_lora_params_by_config(self, adapter_name: str, target_regex: str):
        """
        直接從 transformer.named_parameters() 收集 LoRA 參數：
        - 抓到 *.lora_A/B/up/down.weight
        - 用 target_regex（你的 target_modules）對 base 路徑做白名單篩選
        """
        pat = re.compile(target_regex)
        suffixes = (".lora_A.weight", ".lora_B.weight", ".lora_up.weight", ".lora_down.weight")

        selected = []
        total_lora = 0
        for name, p in self.transformer.named_parameters():
            if any(sfx in name for sfx in suffixes):
                total_lora += 1
                base = name
                for sfx in suffixes:
                    base = base.replace(sfx, "")
                # e.g. base = "transformer.blocks.3.attn.to_q"
                if pat.search(base):
                    p.requires_grad_(True)
                    selected.append(p)

        if not selected:
            # regex 太嚴了 → 回退：把所有 LoRA 參數全開，避免 optimizer 空參數
            for name, p in self.transformer.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)
                    selected.append(p)
            print("[LoRA] WARN: target_modules regex 沒匹配到任何層，已回退為啟用所有 LoRA 權重。")

        print(f"[LoRA] Trainable params matched by regex: {len(selected)} / {total_lora}")
        return selected
    def init_lora(self, lora_path: str, lora_config: dict):
        assert (lora_path is not None) or (lora_config is not None), \
            "Either lora_path or lora_config must be provided."
        adapter_name = self.active_adapter
        # 預設：用 config 的 target_modules 正則；若無則退回 '.*'
        target_regex = (lora_config or {}).get("target_modules", ".*")

        if lora_path:
            local_dir, weight_name = self._resolve_local_lora(lora_path)

            if local_dir is not None:
                # 本地載入
                self.flux_pipe.load_lora_weights(local_dir, weight_name=weight_name, adapter_name=adapter_name)
                print(f"[LoRA] Loaded from local: {os.path.join(local_dir, weight_name)} "
                      f"as adapter='{adapter_name}'")
                # safep = os.path.join(local_dir, "k_mlp.safetensors")

                # state = safetensors_load(safep)
                # self.k_encoder.load_state_dict(state, strict=True)
                # print(f"[K-MLP] Loaded from {safep}")

            else:
                # HF repo（可留著以後用）
                self.flux_pipe.load_lora_weights(lora_path, weight_name=None, adapter_name=adapter_name)
                print(f"[LoRA] Loaded from HF repo: {lora_path} as adapter='{adapter_name}'")

            # 僅開啟符合 target_regex 的 LoRA 權重
            lora_layers = self._collect_lora_params_by_config(adapter_name, target_regex)
        else:
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        # ---- 無論 有沒有load，K-MLP 一律可訓練並併入回傳清單 ----
        #k_params = list(self.k_encoder.parameters())
        #不用先把grad打開，之後configure_optimizers會一併把lora_layer裡面的東西打開
        # for p in k_params:
        #     p.requires_grad_(True)

        return list(lora_layers)#+k_params

    def save_lora(self, path: str):
        # different branch use different adapter names: I use "default"
        os.makedirs(path, exist_ok=True)
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )
        # Save K-MLP if it exists
        if hasattr(self, 'k_encoder'):
            k_path_st = os.path.join(path, "k_mlp.safetensors")
            state = self.k_encoder.state_dict()
            safetensors_save(state, k_path_st)
            print(f"[K-MLP] Saved to {k_path_st}")

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer

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

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
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
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        branch_n = 2 + len(conditions)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device)
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
                "batch_idx": batch_idx,
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
                batch
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
    wandb_config = training_config.get("wandb", None)
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
        multiprocessing_context="spawn",
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
