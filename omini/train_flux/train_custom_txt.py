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
        img = img.resize((self.size, self.size), Image.BICUBIC)
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        return {
            "image": self.to_tensor(img),
            "description": text,
        }


@torch.no_grad()
# def test_function(model, save_path, file_name):
#     """Simple text->image sampling using one random json prompt."""
#     if not hasattr(model, "flux_pipe"):
#         return
#     # Pick a dummy prompt if no data hook is wired; caller can override.
#     prompt = "a test image"
#     os.makedirs(save_path, exist_ok=True)
#     out = model.flux_pipe(prompt=prompt, height=512, width=512)
#     img = out.images[0]
#     img.save(os.path.join(save_path, f"{file_name}.png"))
def test_function(model, save_path, file_name):
    """Simple text->image sampling using one random json prompt."""
    if not hasattr(model, "flux_pipe"):
        return
    #model.set_adapters(["default"])
    # Load 1.txt from training dataset
    txt_path = "/home/chchen/lab/flowchart_dataset/structures/1.txt"
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        print(f"[Test] Failed to load {txt_path}: {e}")
        text = "a test image"

    prompt = text
    print(f"[Test] Using prompt from 1.txt (length: {len(prompt)} chars)")
    
    os.makedirs(save_path, exist_ok=True)
    out = model.flux_pipe(prompt=prompt, height=512, width=512)
    img = out.images[0]
    img.save(os.path.join(save_path, f"{file_name}.png"))

def main():
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
