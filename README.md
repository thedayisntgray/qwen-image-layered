# Qwen-Image-Layered API

[![RunPod](https://api.runpod.io/badge/username/qwen-image-layered)](https://console.runpod.io/hub/username/qwen-image-layered)

[![One-Click Pod Deployment](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/685b44aed6fc50d169003af4_banner-runpod.webp)](https://console.runpod.io/deploy?template=YOUR_TEMPLATE_ID)

A production-ready RunPod serverless endpoint for Alibaba's Qwen-Image-Layered model - an advanced image decomposition model that breaks down images into multiple RGBA layers for independent manipulation and high-fidelity editing.

## Features

- **Layer Decomposition** - Decomposes images into 3-8 RGBA layers
- **Independent Manipulation** - Each layer can be edited separately (resize, reposition, recolor)
- **GPU Optimized** - Runs on A100 80GB, H100 PCIe, H100 HBM3, H100 NVL, and high-end workstation GPUs
- **Auto-scaling** - Scales to 0 when idle to save costs
- **Network Volume Storage** - Model cached persistently across all workers
- **Fast Cold Starts** - Optimized Docker image with pre-installed dependencies

## Model Specifications

- **Model**: `Qwen/Qwen-Image-Layered`
- **Recommended VRAM**: 80GB (A100/H100 recommended)
- **Precision**: bfloat16 (CUDA) / float32 (CPU)
- **Recommended Resolution**: 640x640
- **Output**: Multiple RGBA layers with transparency
- **License**: Check official repository

## API Usage

### Input Format

```json
{
  "input": {
    "image": "base64_encoded_image_data",
    "layers": 4,
    "prompt": "optional text prompt for guided decomposition",
    "negative_prompt": "",
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "resolution": 640,
    "cfg_normalize": true,
    "use_en_prompt": true,
    "seed": null
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | **required** | Base64 encoded input image |
| `layers` | integer | `4` | Number of layers to decompose into (3-8 recommended) |
| `prompt` | string | `null` | Optional text prompt for guided decomposition |
| `negative_prompt` | string | `""` | What to avoid in decomposition |
| `num_inference_steps` | integer | `50` | Number of denoising steps |
| `true_cfg_scale` | float | `4.0` | Classifier-free guidance scale |
| `resolution` | integer | `640` | Processing resolution (640 recommended) |
| `cfg_normalize` | boolean | `true` | Enable CFG normalization |
| `use_en_prompt` | boolean | `true` | Use English prompts |
| `seed` | integer | `null` | Random seed for reproducibility |

### Output Format

```json
{
  "layers": [
    {
      "image": "base64_encoded_rgba_png",
      "mode": "RGBA",
      "size": [640, 640]
    },
    {
      "image": "base64_encoded_rgba_png",
      "mode": "RGBA",
      "size": [640, 640]
    }
  ],
  "seed": 12345,
  "num_layers": 4
}
```

### Example Request (Python)

```python
import runpod
import base64
from PIL import Image
import io

runpod.api_key = "your_api_key_here"

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Load and encode input image
with open("input.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

request = {
    "input": {
        "image": image_b64,
        "layers": 4,
        "resolution": 640,
        "num_inference_steps": 50,
        "seed": 42
    }
}

run = endpoint.run_sync(request)

# Save each layer
for i, layer in enumerate(run['layers']):
    img_data = base64.b64decode(layer['image'])
    image = Image.open(io.BytesIO(img_data))
    image.save(f'layer_{i}.png')
    print(f"Saved layer {i}: {layer['mode']} {layer['size']}")

print(f"Decomposed into {run['num_layers']} layers with seed: {run['seed']}")
```

### Example Request (cURL)

```bash
# First encode your image to base64
base64 -i input.jpg -o image.b64

# Then make the request
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "'$(cat image.b64)'",
      "layers": 4,
      "resolution": 640,
      "num_inference_steps": 50
    }
  }'
```

## Alternative API Server

The repository includes `runpod_startup.sh` which sets up a FastAPI server with additional endpoints:

- `POST /decompose` - Synchronous layer decomposition
- `POST /decompose_async` - Asynchronous decomposition with job tracking
- `POST /decompose_file` - Direct file upload endpoint
- `GET /status/{job_id}` - Check async job status
- `GET /result/{job_id}` - Get async job results

## Deployment Configuration

The template is configured with optimal settings in `runpod.toml`:

- **GPU Types**: A100 80GB PCIe, H100 PCIe, H100 HBM3, H100 NVL, RTX 6000 Blackwell, RTX 6000 Blackwell Workstation, RTX Pro 6000 Max-Q Workstation
- **Minimum VRAM**: 80GB
- **Container Disk**: 5GB (code + dependencies)
- **Network Volume**: ~100GB (persistent model storage) - **⚠️ REQUIRED**
- **Workers**: 0-3 (auto-scaling)
- **Timeout**: 600 seconds per job

### ⚠️ Important: Network Volume Required

**You MUST attach a network volume (~100GB) when deploying this endpoint.**

The Qwen-Image-Layered model requires significant disk space. Without a network volume:
- ❌ Deployment will fail due to insufficient disk space
- ❌ Model cannot be downloaded or cached
- ❌ Workers will crash during initialization

The network volume:
- ✅ Stores the model persistently across all workers
- ✅ Prevents re-downloading the model on every cold start
- ✅ Enables faster scaling and startup times

## Use Cases

1. **Image Editing** - Modify individual elements without affecting others
2. **Background Removal** - Separate foreground and background layers
3. **Object Manipulation** - Move, resize, or recolor specific objects
4. **Creative Composition** - Rearrange layers for new compositions
5. **Animation** - Use layers for frame-by-frame animation
6. **Design Workflows** - Export to design tools with layer support

## Performance

- **Cold Start**: ~60-120 seconds (model download on first run)
- **Warm Inference**: ~30-60 seconds (depends on steps and layer count)
- **Memory Usage**: ~50-70GB VRAM depending on resolution

## Tips for Best Results

1. **Resolution**: Use 640x640 for optimal quality/speed balance
2. **Layer Count**: 3-5 layers work well for most images
3. **Complex Images**: Use more layers (6-8) for complex scenes
4. **Prompts**: Use prompts for guided decomposition when needed
5. **Post-Processing**: Layers are RGBA - perfect for further editing

## License

This endpoint uses the Qwen-Image-Layered model. For licensing information, visit the [official Qwen-Image-Layered repository](https://huggingface.co/Qwen/Qwen-Image-Layered).

## Support

For issues or questions about this RunPod template, please open an issue on the GitHub repository.