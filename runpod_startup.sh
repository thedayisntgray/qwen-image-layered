#!/bin/bash
set -e

echo "🚀 Qwen-Image-Layered API Setup"
echo "================================"

# Clean disk space before model download
echo "🧹 Cleaning disk space..."
rm -rf /workspace/.cache/huggingface/hub/models--* 2>/dev/null || true
rm -rf /root/.cache/huggingface/* 2>/dev/null || true
rm -rf /tmp/* 2>/dev/null || true
pip cache purge 2>/dev/null || true
echo "📊 Disk space: $(df -h /workspace | tail -1 | awk '{print $4}') available"

# Install dependencies (official Qwen-Image-Layered setup)
echo "📦 Installing dependencies..."
python3 -m pip install --upgrade pip

# Core dependencies with specific versions for Qwen-Image-Layered
python3 -m pip install git+https://github.com/huggingface/diffusers
python3 -m pip install transformers>=4.51.3 accelerate safetensors
python3 -m pip install hf-transfer

# API dependencies
python3 -m pip install fastapi uvicorn pillow python-pptx requests

# Clean cache after install
python3 -m pip cache purge
echo "✅ Dependencies installed!"
echo "📊 PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "📊 Diffusers: $(python3 -c 'import diffusers; print(diffusers.__version__)')"
echo "📊 Transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"
echo "📊 Accelerate: $(python3 -c 'import accelerate; print(accelerate.__version__)')"
echo "📊 HF Transfer: $(python3 -c 'import hf_transfer; print(hf_transfer.__version__)')"

# Create minimal API server
echo "🔧 Creating API server..."
cat > /workspace/qwen_api.py << 'EOF'
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from diffusers import QwenImageLayeredPipeline
import torch
import uvicorn
from PIL import Image
import base64
import io
import asyncio
import uuid
from typing import Optional, List, Dict
import json

app = FastAPI(title="Qwen-Image-Layered API")
pipeline = None
jobs = {}  # In-memory job store

@app.on_event("startup")
async def startup():
    global pipeline
    print("🚀 Loading Qwen-Image-Layered model...")
    
    model_name = "Qwen/Qwen-Image-Layered"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipeline = QwenImageLayeredPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipeline = pipeline.to(device)
    
    print(f"✅ Model loaded on {device}")
    print(f"📊 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

class LayerDecomposeRequest(BaseModel):
    image: str  # base64 encoded image
    layers: int = 4  # number of layers to decompose into
    prompt: Optional[str] = None  # optional prompt for guided decomposition
    negative_prompt: Optional[str] = ""
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    resolution: int = 640  # recommended resolution
    cfg_normalize: bool = True
    use_en_prompt: bool = True
    seed: Optional[int] = None

class Layer(BaseModel):
    image: str  # base64 encoded RGBA image
    mode: str  # image mode (should be RGBA)
    size: List[int]  # [width, height]

class LayerDecomposeResponse(BaseModel):
    layers: List[Layer]
    seed: int
    num_layers: int

class JobStatus(BaseModel):
    job_id: str
    status: str
    detail: Optional[str] = None

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL image"""
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

def decompose_image(request: LayerDecomposeRequest):
    """Synchronous image layer decomposition"""
    # Convert base64 to PIL image
    input_image = base64_to_image(request.image)
    
    generator = None
    if request.seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(request.seed)
    
    with torch.inference_mode():
        kwargs = {
            "image": input_image,
            "num_inference_steps": request.num_inference_steps,
            "true_cfg_scale": request.true_cfg_scale,
            "generator": generator,
            "layers": request.layers,
            "resolution": request.resolution,
            "cfg_normalize": request.cfg_normalize,
            "use_en_prompt": request.use_en_prompt
        }
        
        if request.prompt:
            kwargs["prompt"] = request.prompt
            kwargs["negative_prompt"] = request.negative_prompt
        
        result = pipeline(**kwargs)
    
    # Convert layers to response format
    layers = []
    for layer in result.images:
        layers.append(Layer(
            image=image_to_base64(layer),
            mode=layer.mode,
            size=list(layer.size)
        ))
    
    used_seed = request.seed if request.seed is not None else (generator.initial_seed() if generator else 0)
    
    return layers, used_seed

async def run_decompose_job(job_id: str, request: LayerDecomposeRequest):
    """Async job runner"""
    jobs[job_id] = {"status": "running"}
    try:
        print(f"🎨 Job {job_id}: Decomposing image into {request.layers} layers...")
        layers, used_seed = decompose_image(request)
        jobs[job_id] = {"status": "done", "layers": layers, "seed": used_seed}
        print(f"✅ Job {job_id}: Complete! Generated {len(layers)} layers")
    except Exception as e:
        jobs[job_id] = {"status": "error", "detail": str(e)}
        print(f"❌ Job {job_id}: Failed - {e}")

@app.get("/")
async def root():
    return {
        "message": "Qwen-Image-Layered API", 
        "docs": "/docs",
        "description": "Decompose images into multiple RGBA layers for independent manipulation"
    }

@app.get("/health")
async def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "Qwen/Qwen-Image-Layered"}

@app.post("/decompose", response_model=LayerDecomposeResponse)
async def decompose(request: LayerDecomposeRequest):
    """Synchronous layer decomposition (use for small images/steps)"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    print(f"🎨 Decomposing image into {request.layers} layers...")
    layers, used_seed = decompose_image(request)
    print("✅ Decomposed successfully!")
    return LayerDecomposeResponse(layers=layers, seed=used_seed, num_layers=len(layers))

@app.post("/decompose_async", response_model=JobStatus)
async def decompose_async(request: LayerDecomposeRequest):
    """Async layer decomposition (recommended for large images/many steps)"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued"}
    asyncio.create_task(run_decompose_job(job_id, request))
    return JobStatus(job_id=job_id, status="queued")

@app.post("/decompose_file", response_model=LayerDecomposeResponse)
async def decompose_file(
    file: UploadFile = File(...),
    layers: int = 4,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = "",
    num_inference_steps: int = 50,
    true_cfg_scale: float = 4.0,
    resolution: int = 640,
    cfg_normalize: bool = True,
    use_en_prompt: bool = True,
    seed: Optional[int] = None
):
    """Upload an image file directly for decomposition"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read and encode the uploaded file
    contents = await file.read()
    img_b64 = base64.b64encode(contents).decode()
    
    # Create request
    request = LayerDecomposeRequest(
        image=img_b64,
        layers=layers,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        resolution=resolution,
        cfg_normalize=cfg_normalize,
        use_en_prompt=use_en_prompt,
        seed=seed
    )
    
    print(f"🎨 Decomposing uploaded image into {layers} layers...")
    layers_result, used_seed = decompose_image(request)
    print("✅ Decomposed successfully!")
    return LayerDecomposeResponse(layers=layers_result, seed=used_seed, num_layers=len(layers_result))

@app.get("/status/{job_id}", response_model=JobStatus)
async def job_status(job_id: str):
    """Check job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    entry = jobs[job_id]
    return JobStatus(job_id=job_id, status=entry["status"], detail=entry.get("detail"))

@app.get("/result/{job_id}", response_model=LayerDecomposeResponse)
async def job_result(job_id: str):
    """Get job result"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    entry = jobs[job_id]
    if entry["status"] != "done":
        raise HTTPException(status_code=202, detail=f"Job status: {entry['status']}")
    return LayerDecomposeResponse(layers=entry["layers"], seed=entry["seed"], num_layers=len(entry["layers"]))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

echo "✅ Setup complete!"
echo "🚀 Starting API server..."
python3 /workspace/qwen_api.py