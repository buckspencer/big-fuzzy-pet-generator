from typing import Optional
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
import json
import io

class PetRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 512
    height: Optional[int] = 512

class PetResponse(BaseModel):
    description: dict
    image: str  # Base64 encoded image

class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Load text generation model (we'll use a fine-tuned model)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            "gpt2",  # Replace with your fine-tuned model
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Load image generation model
        self.image_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

    def generate_description(self, prompt: str) -> dict:
        """Generate a structured description of the royal pet"""
        # Format the prompt for the model
        formatted_prompt = f"Create a detailed description of a royal pet: {prompt}\n"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.text_model.device)
        
        # Generate text
        outputs = self.text_model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Parse the generated text into structured JSON
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # TODO: Implement proper JSON parsing from generated text
        # This is a placeholder structure
        return {
            "name": "Generated Name",
            "species": "Cat/Dog/etc",
            "magical_abilities": ["Ability 1", "Ability 2"],
            "personality": "Description",
            "royal_title": "Title",
            "backstory": generated_text
        }

    def generate_image(self, prompt: str, negative_prompt: str, **kwargs) -> Image.Image:
        """Generate an image of the royal pet"""
        # Add style prompts for consistent royal/magical theme
        enhanced_prompt = f"Highly detailed digital art of a royal magical pet, {prompt}, fantasy style, professional illustration"
        enhanced_negative = f"{negative_prompt}, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, ugly, disgusting, amputation"
        
        # Generate the image
        image = self.image_pipe(
            prompt=enhanced_prompt,
            negative_prompt=enhanced_negative,
            num_inference_steps=kwargs.get("num_inference_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            width=kwargs.get("width", 512),
            height=kwargs.get("height", 512)
        ).images[0]
        
        return image

    def predict(self, request: PetRequest) -> PetResponse:
        """Generate both description and image for the royal pet"""
        # Generate description
        description = self.generate_description(request.prompt)
        
        # Generate image
        image = self.generate_image(
            request.prompt,
            request.negative_prompt or "",
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height
        )
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        import base64
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return PetResponse(
            description=description,
            image=image_base64
        ) 