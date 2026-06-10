
from diffusers import StableDiffusionPipeline
import torch
import os


model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)


device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipe.to(device)

print(f"Model loaded on: {device}")

os.makedirs("outputs", exist_ok=True)
print("Output folder ready")


prompts = [
    ("healthy_ocean", "a thriving coral reef, underwater, vibrant colors, diverse marine life, photorealistic, 8k"),
    ("dying_ocean", "a bleached dead coral reef, underwater, grey and white, eerie, no fish, photorealistic, 8k"),
    ("healthy_forest", "a lush green Amazon rainforest, misty, full of wildlife, golden hour, photorealistic"),
    ("deforested", "a deforested Amazon rainforest, barren land, tree stumps, smoke, dramatic sky, photorealistic"),
    ("flooded_city", "a coastal city flooded by rising sea levels, abandoned buildings, dramatic, cinematic lighting"),
]

for filename, prompt in prompts:
    print(f"Generating: {filename}...")
    
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        negative_prompt="blurry, low quality, cartoon, painting, drawing"
    ).images[0]
    
    image.save(f"outputs/{filename}.png")
    print(f"Saved: outputs/{filename}.png")

print("All images generated!")


