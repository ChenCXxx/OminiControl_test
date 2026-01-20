import argparse
import os
from pathlib import Path
import yaml

# Respect CUDA_VISIBLE_DEVICES if user sets it in shell (same as train.sh)

import torch
from diffusers.pipelines import FluxPipeline
from PIL import Image

from omini.train_flux.trainer import encode_prompt_with_long_text_support


def _load_config(path_hint: str | None) -> dict:
    if path_hint and Path(path_hint).exists():
        with open(path_hint, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    default = Path(__file__).parent / "train" / "config" / "config.yaml"
    if default.exists():
        with open(default, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def _resolve_lora(path_or_dir: str) -> tuple[str, str]:
    path = Path(path_or_dir)
    if path.is_file() and path.suffix == ".safetensors":
        return str(path.parent), path.name
    if path.is_dir():
        candidates = [p.name for p in path.glob("*.safetensors")]
        if not candidates:
            raise FileNotFoundError(f"No .safetensors found under: {path}")
        weight = "default.safetensors" if "default.safetensors" in candidates else candidates[0]
        return str(path), weight
    raise FileNotFoundError(f"Cannot resolve LoRA from: {path_or_dir}")


def _str_to_dtype(name: str | None) -> torch.dtype:
    if not name:
        return torch.float16
    name = name.lower()
    if hasattr(torch, name):
        return getattr(torch, name)
    return torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX with a chosen LoRA ckpt and txt prompt")
    parser.add_argument("--ckpt", required=True, help="Path to LoRA .safetensors file or its folder")
    parser.add_argument("--text", required=True, help="Path to prompt .txt file")
    parser.add_argument("--output", default="inference.png", help="Where to save the generated image")
    parser.add_argument("--match-image", default=None, help="Optional reference image; use its size (floored to /16) for height/width")
    parser.add_argument("--config", default=os.environ.get("OMINI_CONFIG"), help="Config yaml (defaults to train/config/config.yaml)")
    parser.add_argument("--adapter-name", default="default", help="Adapter name used when loading LoRA")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default=None, help="torch dtype name, e.g. bfloat16/float16")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=28, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--prompt", default=None, help="Override prompt text instead of reading the txt file")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = _load_config(args.config)

    base_model = cfg.get("flux_path", "black-forest-labs/FLUX.1-dev")
    dtype = _str_to_dtype(args.dtype or cfg.get("dtype"))
    device_str = args.device
    if device_str.startswith("cuda"):
        # Match training behavior: CUDA_VISIBLE_DEVICES remaps to cuda:0
        device = torch.device("cuda:0")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    else:
        device = torch.device(device_str)
    
    pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe = pipe.to(device)

    lora_dir, weight_name = _resolve_lora(args.ckpt)
    pipe.load_lora_weights(lora_dir, weight_name=weight_name, adapter_name=args.adapter_name)
    pipe.set_adapters(args.adapter_name)

    prompt_text = args.prompt or Path(args.text).read_text(encoding="utf-8").strip()

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    model_cfg = cfg.get("model", {})
    use_chunked = model_cfg.get("use_chunked_text_encoding", False)
    max_seq_len = model_cfg.get("max_sequence_length", 512)
    chunk_overlap = model_cfg.get("text_chunk_overlap", 50)

    if args.match_image:
        ref = Image.open(args.match_image).convert("RGB")
        w, h = ref.size
        args.width = (w // 16) * 16
        args.height = (h // 16) * 16
        print(f"Match image size: {w}x{h} -> use {args.width}x{args.height}")

    token_count = len(pipe.tokenizer_2.encode(prompt_text))
    use_long_text_path = use_chunked and token_count > max_seq_len

    if use_long_text_path:
        prompt_embeds, pooled_embeds, _ = encode_prompt_with_long_text_support(
            flux_pipe=pipe,
            prompt=prompt_text,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_seq_len,
            use_chunked_encoding=use_chunked,
            chunk_overlap=chunk_overlap,
        )
        result = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )
    else:
        result = pipe(
            prompt=prompt_text,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )

    image = result.images[0]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to {output_path}")


if __name__ == "__main__":
    main()
