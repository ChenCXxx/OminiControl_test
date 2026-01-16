import json
import os
import random

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .trainer import OminiModel, get_config, train


class CustomDataset(Dataset):
    """Map (txt content) -> (target image) pairs.

    Expects matching filenames under image_dir/*.png and txt_dir/*.txt.
    Only loads training samples (0-49).
    """

    def __init__(self, image_dir: str, txt_dir: str, size: int = 512, train_indices: list = None):
        self.image_dir = image_dir
        self.txt_dir = txt_dir
        self.size = size
        self.to_tensor = T.ToTensor()

        # 如果沒有指定 train_indices，預設使用 0-49
        if train_indices is None:
            train_indices = list(range(0, 50))  # 0-49 作為訓練資料
        
        pairs = []
        for idx in train_indices:
            # 檢查 .png, .jpg, .jpeg
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                potential_path = os.path.join(self.image_dir, f"{idx}{ext}")
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            
            if img_path is None:
                print(f"[Warning] Image not found for index {idx}, skipping...")
                continue
            
            txt_path = os.path.join(self.txt_dir, f"{idx}.txt")
            if not os.path.exists(txt_path):
                print(f"[Warning] Text not found for index {idx}, skipping...")
                continue
            
            pairs.append((img_path, txt_path))
        
        if not pairs:
            raise RuntimeError(f"No image/txt pairs found for indices {train_indices}")
        
        self.pairs = pairs
        print(f"[Dataset] Loaded {len(self.pairs)} training samples (indices: {min(train_indices)}-{max(train_indices)})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, txt_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        # custom size
        w, h = img.size
        
        new_w = (w // 16) * 16
        new_h = (h // 16) * 16
        if(new_w != w or new_h != h):
            img = img.resize((new_w, new_h), Image.BICUBIC)

        # fix image size
        # img = img.resize((self.size, self.size), Image.BICUBIC)
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        return {
            "image": self.to_tensor(img),
            "description": text,
        }


@torch.no_grad()
def test_function(model, save_path, file_name, batch=None):
    """測試函數：使用獨立的測試集 (9995-9999) 來生成圖片"""
    if not hasattr(model, "flux_pipe"):
        return
    
    # 從配置中獲取資料路徑
    config = get_config()
    ds_cfg = config["train"].get("dataset", {})
    image_dir = ds_cfg.get("image_dir", "images")
    txt_dir = ds_cfg.get("txt_dir", "outputs")
    
    # 測試集索引：9995-9999 (5個測試樣本)
    test_indices = [9995, 9996, 9997, 9998, 9999]
    
    # 隨機選一個測試樣本
    test_idx = random.choice(test_indices)
    
    # 讀取測試圖片和文本
    img_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        potential_path = os.path.join(image_dir, f"{test_idx}{ext}")
        if os.path.exists(potential_path):
            img_path = potential_path
            break
    
    txt_path = os.path.join(txt_dir, f"{test_idx}.txt")
    
    if img_path is None or not os.path.exists(txt_path):
        print(f"[Test] Warning: Test sample {test_idx} not found, using batch data instead")
        # 如果測試資料不存在，退回使用 batch 資料
        if batch is not None:
            imgs = batch["image"]
            prompt = batch["description"][0]
            target_h, target_w = imgs.shape[2], imgs.shape[3]
            print(f"[Test] Using batch image size: {target_w}x{target_h}")
        else:
            target_h, target_w = 512, 512
            prompt = "a test image"
            print(f"[Test] Using default prompt")
    else:
        # 成功讀取測試資料
        with open(txt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        # 讀取圖片尺寸作為參考
        img = Image.open(img_path).convert("RGB")
        target_w, target_h = img.size
        target_w = (target_w // 16) * 16
        target_h = (target_h // 16) * 16
        
        print(f"[Test] Using test sample {test_idx}: size {target_w}x{target_h}, prompt length: {len(prompt)} chars")
        print(f"[Test] Prompt preview: {prompt[:80]}...")

    # 獲取模型的長文本處理配置（與 training 保持一致）
    model_config = getattr(model, 'model_config', {})
    max_seq_len = model_config.get('max_sequence_length', 512)
    use_chunked = model_config.get('use_chunked_text_encoding', False)
    
    # 檢查 prompt 是否過長
    tokenizer = model.flux_pipe.tokenizer_2
    token_count = len(tokenizer.encode(prompt))
    if token_count > max_seq_len:
        print(f"[Test] WARNING: Prompt has {token_count} tokens (max={max_seq_len}), will be {'chunked' if use_chunked else 'truncated'}")
    
    # generate and save
    os.makedirs(save_path, exist_ok=True)
    
    # 使用與 training 相同的文本編碼策略
    if use_chunked and token_count > max_seq_len:
        # 使用與 training_step 相同的長文本處理
        from .trainer import encode_prompt_with_long_text_support
        
        prompt_embeds, pooled_embeds, text_ids = encode_prompt_with_long_text_support(
            flux_pipe=model.flux_pipe,
            prompt=prompt,
            device=model.flux_pipe.device,
            num_images_per_prompt=1,
            max_sequence_length=max_seq_len,
            use_chunked_encoding=use_chunked,
            chunk_overlap=model_config.get('text_chunk_overlap', 50),
        )
        
        # 使用預編碼的 embeddings 生成圖片
        out = model.flux_pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            height=target_h,
            width=target_w,
        )
        
        # 512x512 測試
        out_2 = model.flux_pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            height=512,
            width=512,
        )
    else:
        # 標準模式：直接使用 prompt（會自動截斷）
        out = model.flux_pipe(prompt=prompt, height=target_h, width=target_w)
        out_2 = model.flux_pipe(prompt=prompt, height=512, width=512)
    
    img = out.images[0]
    # 檔名包含測試樣本編號（如果有的話）
    sample_suffix = f"_test{test_idx}" if 'test_idx' in locals() else ""
    img.save(os.path.join(save_path, f"{file_name}{sample_suffix}.png"))
    
    # 512x512 測試
    img_2 = out_2.images[0]
    img_2.save(os.path.join(save_path, f"{file_name}{sample_suffix}_512x512.png"))
    
    print(f"[Test] Saved images to {save_path}/{file_name}{sample_suffix}.png")

def main():
    import warnings
    from transformers import logging as hf_logging
    from diffusers import logging as df_logging

    warnings.filterwarnings("ignore", message=".*77 tokens.*")
    warnings.filterwarnings("ignore", message=".*max_sequence_length.*")
    hf_logging.set_verbosity_error()
    df_logging.set_verbosity_error()
    # 或更精準地用 category=UserWarning
    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        torch.cuda.set_device(0)
    
    # Initialize custom dataset from config
    ds_cfg = training_config.get("dataset", {})
    image_dir = ds_cfg.get("image_dir", "images")
    txt_dir = ds_cfg.get("txt_dir", "structure")
    target_size = ds_cfg.get("target_size", 512)
    dataset = CustomDataset(image_dir=image_dir, txt_dir=txt_dir, size=target_size)

    # debugging
    try:
        first = dataset[0]
        print(f"[Dataset] first description length (chars): {len(first['description'])}")
    except Exception as e:
        print(f"[Dataset] failed to read first sample: {e}")

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        adapter_names=["default", "default"],
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
