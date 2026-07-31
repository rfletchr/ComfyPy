# ComfyPy

Python client for the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) API. HTTP and WebSocket, sync, stdlib where possible.

```python
from comfypy import ComfyClient, ExecutionStarted, NodeProgress, ExecutionFinished

client = ComfyClient()

prompt = {
    "4": {
        "inputs": {"ckpt_name": "epicrealismXL_pureFix.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint - BASE"},
    },
    "5": {
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        "class_type": "EmptyLatentImage",
        "_meta": {"title": "Empty Latent Image"},
    },
    "6": {
        "inputs": {
            "text": "evening sunset scenery blue sky nature, glass bottle with a galaxy in it",
            "clip": ["4", 1],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "7": {
        "inputs": {"text": "text, watermark", "clip": ["4", 1]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "17": {
        "inputs": {"samples": ["53", 0], "vae": ["4", 2]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "19": {
        "inputs": {"filename_prefix": "ComfyUI", "images": ["17", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
    "53": {
        "inputs": {
            "seed": 269671452348788,
            "steps": 20,
            "cfg": 8,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
}

result = client.post_prompt(prompt)

for event in client.iter_messages():
    if isinstance(event, ExecutionStarted):
        print("running…")
    elif isinstance(event, NodeProgress):
        print(f"  {event.value}/{event.max}")
    elif isinstance(event, ExecutionFinished):
        break

print(client.get_history_prompt(result["prompt_id"]))
```