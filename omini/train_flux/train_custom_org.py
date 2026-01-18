import json
import os
import random

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .trainer import OminiModel, get_config, train


class CustomDataset(Dataset):
    """Map (json text) -> (target image) pairs.

    Expects matching filenames under image_dir/*.png and json_dir/*.json.
    """

    def __init__(self, image_dir: str, json_dir: str, size: int = 512):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.size = size
        self.to_tensor = T.ToTensor()

        img_files = [f for f in os.listdir(self.image_dir) if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}]
        imgs = sorted([os.path.join(self.image_dir, f) for f in img_files])
        pairs = []
        for img_path in imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            jp = os.path.join(self.json_dir, f"{stem}.json")
            if not os.path.exists(jp):
                raise FileNotFoundError(f"Missing json for {os.path.basename(img_path)} at {jp}")
            pairs.append((img_path, jp))
        if not pairs:
            raise RuntimeError("No image/json pairs found.")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, json_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.size, self.size), Image.BICUBIC)
        with open(json_path, "r") as f:
            obj = json.load(f)
        text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

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
    # Load 1.json from training dataset
    json_path = "/home/chchen/lab/flowchart_dataset/structures/1.json"
    try:
        with open(json_path, "r") as f:
            obj = json.load(f)
        prompt = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        print(f"[Test] Using prompt from 1.json (length: {len(prompt)} chars)")
    except Exception as e:
        print(f"[Test] Failed to load {json_path}: {e}")
        prompt = "a test image"
    
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
    json_dir = ds_cfg.get("json_dir", "structure")
    target_size = ds_cfg.get("target_size", 512)
    dataset = CustomDataset(image_dir=image_dir, json_dir=json_dir, size=target_size)

    # debugging
    try:
        first = dataset[0]
        print(f"[Dataset] first description length (chars): {len(first['description'])}")
        print(f"[Dataset] first description content: {first['description']}")
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
        adapter_names=[None, "default"],
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
