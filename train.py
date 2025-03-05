import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler
)
from datasets import Dataset
import json
from pathlib import Path
import os
from typing import Dict, List
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_checkpoint(model_dir: str) -> str:
    """Get the path to the latest model checkpoint"""
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    return os.path.join(model_dir, max(checkpoints, key=lambda x: int(x.split("-")[1])))

def load_training_data(text_data_path: str) -> Dataset:
    """Load and prepare text training data"""
    with open(text_data_path, 'r') as f:
        data = json.load(f)
    
    # Format data for training
    formatted_data = []
    for item in data:
        text = f"""Create a royal pet description following this exact JSON structure:
{{
    "user_facing": {{
        "name": "{item['user_facing']['name']}",
        "kingdom": "{item['user_facing']['kingdom']}",
        "backstory": "{item['user_facing']['backstory']}"
    }},
    "image_generation": {{
        "species": "{item['image_generation']['species']}",
        "breed": "{item['image_generation']['breed']}",
        "physical_description": "{item['image_generation']['physical_description']}",
        "royal_attire": "{item['image_generation']['royal_attire']}",
        "setting": "{item['image_generation']['setting']}",
        "style_notes": "{item['image_generation']['style_notes']}"
    }}
}}"""
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def train_text_model(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "models/text_model",
    training_data_path: str = "data/text/processed_pets.json",
    num_train_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    resume_from_checkpoint: bool = True
):
    """Fine-tune the text generation model"""
    logger.info("Loading text model and tokenizer...")
    
    # Check for existing checkpoint
    checkpoint = None
    if resume_from_checkpoint:
        checkpoint = get_latest_checkpoint(output_dir)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        else:
            logger.info("No checkpoint found, starting from base model")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and prepare training data
    logger.info("Loading training data...")
    dataset = load_training_data(training_data_path)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,  # Keep only the last 3 checkpoints
        fp16=True,
        gradient_accumulation_steps=4,
        report_to="tensorboard",
        run_name=f"royal_pet_text_{timestamp}"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    # Train the model
    logger.info("Starting text model training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save the model
    logger.info("Saving text model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

def train_image_model(
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    output_dir: str = "models/image_model",
    training_data_path: str = "data/text/processed_pets.json",
    image_dir: str = "data/images",
    num_train_epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 1e-6,
    resume_from_checkpoint: bool = True
):
    """Fine-tune the image generation model"""
    logger.info("Loading image model...")
    
    # Check for existing checkpoint
    checkpoint = None
    if resume_from_checkpoint:
        checkpoint = get_latest_checkpoint(output_dir)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint}")
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                checkpoint,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        else:
            logger.info("No checkpoint found, starting from base model")
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    
    # Load training data
    with open(training_data_path, 'r') as f:
        data = json.load(f)
    
    # Prepare training data
    training_data = []
    for item in data:
        # Find corresponding image
        species = item['image_generation']['species'].lower()
        breed = item['image_generation']['breed'].lower().replace(' ', '_')
        image_path = Path(image_dir) / species / f"{species}_{breed}_001.jpg"
        
        if image_path.exists():
            training_data.append({
                "prompt": f"{item['image_generation']['physical_description']}, {item['image_generation']['setting']}, {item['image_generation']['style_notes']}",
                "image_path": str(image_path)
            })
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training configuration
    training_args = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": 4,
        "fp16": True,
        "report_to": "tensorboard",
        "run_name": f"royal_pet_image_{timestamp}"
    }
    
    # Train the model
    logger.info("Starting image model training...")
    pipeline.train(
        training_data=training_data,
        **training_args
    )
    
    # Save the model
    logger.info("Saving image model...")
    pipeline.save_pretrained(output_dir)

if __name__ == "__main__":
    # Train text model
    train_text_model(resume_from_checkpoint=True)
    
    # Train image model
    train_image_model(resume_from_checkpoint=True) 