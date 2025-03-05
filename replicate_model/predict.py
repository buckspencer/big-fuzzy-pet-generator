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
        # Load text generation model (TinyLlama)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # Load image generation model (Stable Diffusion XL)
        self.image_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

    def generate_description(self, prompt: str) -> dict:
        """Generate a structured description of the royal pet"""
        # Format the prompt for the model
        formatted_prompt = f"""Create a royal pet description following this exact JSON structure:
{{
    "user_facing": {{
        "name": "royal title and name",
        "kingdom": "the animal's kingdom or territory",
        "backstory": "1-2 paragraphs about the royal animal's reign and significance"
    }},
    "image_generation": {{
        "species": "primary species",
        "breed": "specific breed",
        "physical_description": "detailed description for image generation",
        "royal_attire": "description of royal accessories, crown, etc.",
        "setting": "royal throne room, court, or territory",
        "style_notes": "specific style requirements for the image"
    }}
}}

Create a royal pet based on this prompt: {prompt}"""

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.text_model.device)
        
        # Generate text
        outputs = self.text_model.generate(
            **inputs,
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Parse the generated text into JSON
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            # Extract JSON from the generated text
            json_str = generated_text.split("{", 1)[1].rsplit("}", 1)[0]
            json_str = "{" + json_str + "}"
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            # Return a default structure if parsing fails
            return {
                "user_facing": {
                    "name": "Unknown Royal Pet",
                    "kingdom": "The Unknown Kingdom",
                    "backstory": "A mysterious royal pet whose story is yet to be told."
                },
                "image_generation": {
                    "species": "Unknown",
                    "breed": "Unknown",
                    "physical_description": "A majestic creature of unknown origin",
                    "royal_attire": "Royal regalia",
                    "setting": "A grand throne room",
                    "style_notes": "Regal portrait style"
                }
            }

    def generate_image(self, prompt: str, negative_prompt: str, **kwargs) -> Image.Image:
        """Generate an image of the royal pet"""
        # Add style prompts for consistent royal/magical theme
        enhanced_prompt = f"Highly detailed digital art of a royal animal ruler, {prompt}, fantasy style, professional illustration, regal pose, royal throne room, dramatic lighting"
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
        
        # Generate image using the physical description from the generated JSON
        image_prompt = description["image_generation"]["physical_description"]
        setting_prompt = description["image_generation"]["setting"]
        style_prompt = description["image_generation"]["style_notes"]
        
        full_prompt = f"{image_prompt}, {setting_prompt}, {style_prompt}"
        
        image = self.generate_image(
            full_prompt,
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