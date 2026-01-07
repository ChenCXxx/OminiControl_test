import argparse
import os
import torch
from diffusers.pipelines import FluxPipeline

from omini.pipeline.flux_omini import generate as flux_generate


def load_pipe(base_id: str, device: str, dtype: torch.dtype):
    return FluxPipeline.from_pretrained(base_id, torch_dtype=dtype).to(device)


def maybe_load_lora(pipe: FluxPipeline, lora_dir: str | None, adapter: str, weight: str | None):
    if not lora_dir:
        return None
    weight_name = weight or f"{adapter}.safetensors"
    pipe.load_lora_weights(lora_dir, weight_name=weight_name, adapter_name=adapter)
    return [adapter, adapter]  # text branch, image branch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--adapter_name", default="default")
    parser.add_argument("--lora_dir", default=None, help="folder containing safetensors")
    parser.add_argument("--weight_name", default=None, help="filename inside lora_dir; default is <adapter>.safetensors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="output.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    pipe = load_pipe(args.base_id, device=device, dtype=dtype)

    main_adapter = maybe_load_lora(pipe, args.lora_dir, args.adapter_name, args.weight_name)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    result = flux_generate(
        pipeline=pipe,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance,
        main_adapter=main_adapter,
        generator=generator,
        max_sequence_length=pipe.config.get("max_sequence_length", 512),
    )
    img = result.images[0]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img.save(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()