"""
RunPod Serverless Handler for Qwen-Image-Layered
"""
import runpod
from diffusers import QwenImageLayeredPipeline
import torch
from PIL import Image
import base64
import io
import os
from typing import Optional, List, Dict

# Global model instance (loaded once on cold start)
pipeline = None

def load_model():
    """Load model once during cold start"""
    global pipeline
    if pipeline is not None:
        return pipeline

    print("🚀 Loading Qwen-Image-Layered model...")

    model_name = "Qwen/Qwen-Image-Layered"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = QwenImageLayeredPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipeline = pipeline.to(device)

    print(f"✅ Model loaded on {device}")
    print(f"📊 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    return pipeline

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL image"""
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

def generate_layers(job):
    """
    RunPod handler function for Qwen-Image-Layered
    Input format: {"input": {"image": "base64...", "layers": 4, ...}}
    Output format: {"layers": [{"image": "base64...", "alpha": "base64..."}, ...], "seed": 123}
    """
    job_input = job["input"]

    # Get input image
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "image is required (base64 encoded)"}
    
    try:
        input_image = base64_to_image(image_b64)
    except Exception as e:
        return {"error": f"Failed to decode image: {str(e)}"}

    # Get parameters
    layers = job_input.get("layers", 4)  # Number of layers to decompose into
    prompt = job_input.get("prompt", None)  # Optional prompt for guided decomposition
    negative_prompt = job_input.get("negative_prompt", "")
    num_inference_steps = job_input.get("num_inference_steps", 50)
    true_cfg_scale = job_input.get("true_cfg_scale", 4.0)
    resolution = job_input.get("resolution", 640)  # Recommended resolution
    cfg_normalize = job_input.get("cfg_normalize", True)
    use_en_prompt = job_input.get("use_en_prompt", True)
    seed = job_input.get("seed", None)

    print(f"🎨 Decomposing image into {layers} layers...")
    if prompt:
        print(f"📝 Using prompt: {prompt[:100]}...")

    # Load model if not already loaded
    pipe = load_model()

    # Setup generator for seed
    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    # Generate layers
    with torch.inference_mode():
        kwargs = {
            "image": input_image,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "generator": generator,
            "layers": layers,
            "resolution": resolution,
            "cfg_normalize": cfg_normalize,
            "use_en_prompt": use_en_prompt
        }
        
        if prompt:
            kwargs["prompt"] = prompt
            kwargs["negative_prompt"] = negative_prompt
        
        result = pipe(**kwargs)

    # Convert layers to base64
    output_layers = []
    for layer in result.images:
        # Each layer is an RGBA image
        layer_dict = {
            "image": image_to_base64(layer),
            "mode": layer.mode,  # Should be "RGBA"
            "size": list(layer.size)
        }
        output_layers.append(layer_dict)

    used_seed = seed if seed is not None else (generator.initial_seed() if generator else 0)

    print(f"✅ Decomposed successfully! Generated {len(output_layers)} layers, Seed: {used_seed}")

    return {
        "layers": output_layers,
        "seed": used_seed,
        "num_layers": len(output_layers)
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": generate_layers})