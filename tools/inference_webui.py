import gradio as gr
import torch
from diffusers import FluxPipeline
import os
import gc

# Global variable to hold the pipeline
pipe = None
current_base_model = None

def load_model(base_model_path, lora_file, lora_path, device_name, use_cpu_offload):
    global pipe, current_base_model
    
    device = "cuda" if not device_name else device_name
    print(f"Loading Base Model: {base_model_path} on {device}")
    
    # 1. Load Base Model (if not already loaded or if changed)
    if pipe is None or current_base_model != base_model_path:
        try:
            # Clean up old model if exists
            if pipe is not None:
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
            
            print("Initializing FluxPipeline...")
            # Load to CPU first if offloading, otherwise directly to device
            load_device = "cpu" if use_cpu_offload else device
            
            pipe = FluxPipeline.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16
            )
            
            if use_cpu_offload:
                print("Enabling model CPU offload...")
                pipe.enable_model_cpu_offload(device=device)
            else:
                pipe.to(device)
                
            current_base_model = base_model_path
        except Exception as e:
            return f"Error loading base model: {str(e)}"
    else:
        print("Base model already loaded, reusing...")
        # If reusing, we might need to move it if device changed, but for simplicity:
        if not use_cpu_offload:
            pipe.to(device)

    # 2. Load LoRA (if provided)
    # Priority: Server-side Path > Uploaded File
    target_lora = None
    if lora_path and lora_path.strip():
        target_lora = lora_path.strip()
    elif lora_file is not None:
        target_lora = lora_file.name
        
    if target_lora:
        try:
            print(f"Loading LoRA from: {target_lora}")
            # Unload previous LoRA weights if any
            pipe.unload_lora_weights() 
            
            # Load the new LoRA path
            pipe.load_lora_weights(target_lora)
            print("LoRA loaded successfully.")
        except Exception as e:
            return f"Error loading LoRA: {str(e)}"
    
    return "Model loaded successfully!"

def generate_image(prompt, height, width, num_steps, guidance_scale, lora_scale, seed, device_name):
    global pipe
    if pipe is None:
        raise gr.Error("Please load the model first!")
    
    device = "cuda" if not device_name else device_name
    print(f"Generating with Prompt: {prompt}, Size: {width}x{height}, Steps: {num_steps} on {device}")
    
    try:
        # Set seed
        generator = torch.Generator(device).manual_seed(int(seed))
        
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        return image
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

# UI Layout
with gr.Blocks(title="OminiControl/Flux Inference Tool") as demo:
    gr.Markdown("# OminiControl / Flux Inference Tool")
    gr.Markdown("Load a base Flux model and interactively upload a fine-tuned LoRA (.safetensors) to generate images.")
    
    with gr.Row():
        with gr.Column():
            base_model_input = gr.Textbox(
                label="Base Model Path (HuggingFace ID or Local Path)",
                value="black-forest-labs/FLUX.1-dev" 
            )
            with gr.Row():
                device_input = gr.Textbox(
                    label="Device (e.g. cuda:0, cuda:1)",
                    value="cuda:2"
                )
                offload_input = gr.Checkbox(
                    label="Enable CPU Offload (Saves VRAM)",
                    value=False
                )

            with gr.Group():
                lora_path_input = gr.Textbox(
                    label="LoRA Adapter Path (Server-side)",
                    placeholder="/path/to/server/adapter.safetensors",
                    info="If running remotely, enter the absolute path to the .safetensors file on the server."
                )
                lora_file_input = gr.File(
                    label="OR Upload LoRA Adapter",
                    file_types=[".safetensors", ".bin"]
                )
            load_btn = gr.Button("Load Model & Adapter", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3, 
                placeholder="Enter your prompt here..."
            )
            with gr.Row():
                height_input = gr.Number(label="Height", value=512, step=16)
                width_input = gr.Number(label="Width", value=512, step=16)
            with gr.Row():
                steps_input = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=28, step=1)
                guidance_input = gr.Slider(label="Guidance Scale", minimum=0, maximum=10, value=3.5, step=0.1)
                
            lora_scale_input = gr.Slider(label="LoRA Scale", minimum=0, maximum=2.0, value=1.0,  visible=False) # Hidden for now
            seed_input = gr.Number(label="Seed", value=42, precision=0)
            
            generate_btn = gr.Button("Generate Image", variant="primary")
            output_image = gr.Image(label="Generated Image")

    # Event Handlers
    load_btn.click(
        fn=load_model,
        inputs=[base_model_input, lora_file_input, lora_path_input, device_input, offload_input],
        outputs=[status_output]
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input, 
            height_input, 
            width_input, 
            steps_input, 
            guidance_input, 
            lora_scale_input, 
            seed_input,
            device_input
        ],
        outputs=[output_image]
    )

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except KeyboardInterrupt:
        demo.close() # Gracefully close on Ctrl+C
    except Exception as e:
        print(e)
        demo.close()
    
