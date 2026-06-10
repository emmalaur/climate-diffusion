
import os
from diffusers import StableDiffusionPipeline
import torch
import csv


model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipe.to(device)

os.makedirs("outputs", exist_ok=True)

print("-----------------------------------")
print("Climate Change Image Generator")
print("-----------------------------------")
print("Type 'quit' at any prompt to exit.\n")
print("What do you think the future will look like if we don't start taking climate change serious? " )
print("-----------------------------------")


while True:

    print("Where are you? Maybe near a coral reef or on the streets of NYC? ")
    location = input("Location or environment: ").strip()
    if location.lower() == "quit":
        print("Goodbye!")
        break

    print("Do you think the environment will be healthy and thriving or rather flooded or deforested ? ")
    condition = input("Condition: ").strip()
    if condition.lower() == "quit":
        print("Goodbye!")
        break

    print("Is there a center object, human or animal you would like to describe? ")
    centerImage = input("Center Image: ").strip()
    if location.lower() == "quit":
        print("Goodbye!")
        break

    
    style = input("Visual style (e.g. photorealistic, cinematic, aerial drone shot, 8k): ").strip()
    if style.lower() == "quit":
        print("Goodbye!")
        break

    filename = input("Image name (no spaces, no .png): ").strip().replace(" ", "_")
    if filename.lower() == "quit":
        print("Goodbye!")
        break
    if filename == "":
        print("Please enter a name.\n")
        continue
    if os.path.exists(f"outputs/{filename}.png"):
       print(f"Warning: outputs/{filename}.png already exists and will be overwritten.")
       confirm = input("Continue? (y/n): ").strip().lower()
       if confirm != "y":
          continue

    prompt = f"a {condition} {location}, {style}, highly detailed with a {centerImage} in its center."
    negative_prompt = "blurry, low quality, cartoon, painting, drawing"

    print(f"\nGenerated prompt: '{prompt}'")
    print(f"Generating image...")

    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        negative_prompt=negative_prompt
    ).images[0]

    image.save(f"outputs/{filename}.png")
    print(f"Saved: outputs/{filename}.png\n")


with open("outputs/log.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([filename, prompt, style])