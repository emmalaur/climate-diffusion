# Climate Diffusion

Climate change is often discussed through statistics, graphs, and projections. In other Words: Data that is accurate but difficult to feel. This project attempts to close that gap by using AI image generation to make the consequences of climate change visible and immediate.

By answering a few simple questions about a place and its condition, anyone can generate a photorealistic image of what that environment might look like — thriving, or in decline. The goal is not to predict the future precisely, but to make people pause and imagine it.

## How it works

The project uses Stable Diffusion v1.5, a deep learning model that generates images from text prompts. The user is asked four questions:

- What location or environment?
- What condition? 
- What center object
- What visual style? 
- What to name the output file?

These answers are combined into a structured prompt which is passed to the model. The generated image is saved to the `outputs/` folder.

## Requirements

- Python 3.10+
- PyTorch
- Diffusers (HuggingFace)

Install dependencies:

```bash
pip install torch diffusers transformers accelerate
```

## Running the project

```bash
python climate-diffusion.py
```

You will be prompted to describe a scenario. Type `quit` at any point to exit.

## Example

```
What do you think the future will look like?

Location or environment: coral reef
Condition: bleached and dying, no marine life
Center Image: A gray coral
Visual style: underwater, photorealistic, 8k

Generated prompt: 'a bleached and dying, no marine life coral reef, underwater, photorealistic, 8k, highly detailed'
Saved: outputs/dying_reef.png
```

## Model

This project uses [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) via the HuggingFace Diffusers library. The model runs locally on Apple Silicon (MPS) or CPU.

## Credits

Climate Diffusion was created in the span of the UAL course Deep Learning and Big Data Integration: Technologies and Tools 2026.

Author: Emma Reeb
Unit Leader: Xiaowan Yi


