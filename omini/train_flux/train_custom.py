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
    """

    def __init__(self, image_dir: str, txt_dir: str, size: int = 512):
        self.image_dir = image_dir
        self.txt_dir = txt_dir
        self.size = size
        self.to_tensor = T.ToTensor()

        img_files = [f for f in os.listdir(self.image_dir) if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}]
        imgs = sorted([os.path.join(self.image_dir, f) for f in img_files])
        pairs = []
        for img_path in imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            tp = os.path.join(self.txt_dir, f"{stem}.txt")
            if not os.path.exists(tp):
                raise FileNotFoundError(f"Missing txt for {os.path.basename(img_path)} at {tp}")
            pairs.append((img_path, tp))
        if not pairs:
            raise RuntimeError("No image/txt pairs found.")
        self.pairs = pairs

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
# for dataset size = 1
# def test_function(model, save_path, file_name, batch=None, training_config=None):
#     """Simple text->image sampling using one random json prompt."""
#     if not hasattr(model, "flux_pipe"):
#         return
#     # model.set_adapters(["default"])

#     # text prompt (using 1.txt)
#     # txt_path = "/home/chchen/lab/flowchart_dataset/structures/1.txt"
#     # try:
#     #     with open(txt_path, "r", encoding="utf-8") as f:
#     #         prompt = f.read().strip()
#     # except Exception as e:
#     #     print(f"[Test] Failed to load {txt_path}: {e}")
#     #     prompt = "a test image"

#     # image size
#     if batch is not None:
#         imgs = batch["image"]
#         prompt = batch["description"][0]
#         target_h, target_w = imgs.shape[2], imgs.shape[3]  # img shape: (batch_size, channels, height, width)
#         print(f"[Test] Using batch image size: {target_w}x{target_h} and prompt: {prompt[:30]}...")
#     else:
#         target_h, target_w = 512, 512
#         prompt = "a test image"
#         print(f"[Test] Failed to get batch, using default size: {target_w}x{target_h}")

#     # generate and save
#     os.makedirs(save_path, exist_ok=True)
#     out = model.flux_pipe(prompt=prompt, height=target_h, width=target_w)
#     img = out.images[0]
#     img.save(os.path.join(save_path, f"{file_name}.png"))
    
    
#     # for test fix 512x512
#     out_2 = model.flux_pipe(prompt=prompt, height=512, width=512)
#     img_2 = out_2.images[0]
#     img_2.save(os.path.join(save_path, f"{file_name}_512x512.png"))

# for multiple dataset
def test_function(model, save_path, file_name, batch=None, training_config=None):
    if not training_config:
        print("[Test] Error: No training_config provided.")
        return
    
    # get dataset from config
    test_ds_cfg = training_config.get("test_dataset", {})
    image_dir = test_ds_cfg.get("image_dir")
    txt_dir = test_ds_cfg.get("txt_dir")

    if not image_dir or not os.path.exists(image_dir):
        print(f"[Test] Warning: Test image_dir not found: {image_dir}")
        return
    
    current_step_dir = os.path.join(save_path, file_name)
    os.makedirs(current_step_dir, exist_ok=True)
    print(f"[Test] Saving to: {current_step_dir}")

    valid_exts = {".png", ".jpg", ".jpeg"}
    target_files = sorted([f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    
    for img_file in target_files:
        try:
            # load prompt
            stem = os.path.splitext(img_file)[0]
            img_path = os.path.join(image_dir, img_file)
            txt_path = os.path.join(txt_dir, f"{stem}.txt")
            if not os.path.exists(txt_path):
                print(f"[Test] Warning: Missing txt for {img_file}")
                prompt = "a test image"
            else:
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()

            # load image to get size
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            target_w = (w // 16) * 16
            target_h = (h // 16) * 16

            # generate and save
            out = model.flux_pipe(prompt=prompt, height=target_h, width=target_w)
            img = out.images[0]
            img.save(os.path.join(current_step_dir, img_file))
            print(f"[Test] Saved: {img_file} with prompt length {len(prompt)}")

        except Exception as e:
            print(f"[Test] Failed: {e}")

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
