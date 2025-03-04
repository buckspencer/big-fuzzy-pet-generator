import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class RoyalPetDataset(Dataset):
    def __init__(
        self,
        text_data_path: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 512
    ):
        """
        Initialize the Royal Pet Dataset
        
        Args:
            text_data_path: Path to the JSON file containing pet descriptions
            image_dir: Base directory containing pet images
            transform: Optional transforms to apply to images
            image_size: Target size for images (will be resized to square)
        """
        self.image_size = image_size
        self.transform = transform or self._get_default_transforms()
        
        # Load text data
        with open(text_data_path, 'r') as f:
            self.pet_descriptions = json.load(f)
        
        # Setup image paths
        self.image_dir = Path(image_dir)
        self.image_paths = self._get_image_paths()
        
        # Validate data
        self._validate_data()
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transforms"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _get_image_paths(self) -> Dict[str, str]:
        """Map pet descriptions to their corresponding image paths"""
        image_paths = {}
        for pet in self.pet_descriptions:
            species = pet['species'].lower()
            breed = pet['subspecies'].lower().replace(' ', '_')
            # Look for matching image in species directory
            species_dir = self.image_dir / species
            if not species_dir.exists():
                continue
                
            # Try to find matching image
            for img_path in species_dir.glob(f'*{breed}*.jpg'):
                image_paths[pet['name']] = str(img_path)
                break
                
        return image_paths
    
    def _validate_data(self):
        """Validate the dataset"""
        missing_images = []
        for pet in self.pet_descriptions:
            if pet['name'] not in self.image_paths:
                missing_images.append(pet['name'])
        
        if missing_images:
            print(f"Warning: Missing images for pets: {', '.join(missing_images)}")
    
    def __len__(self) -> int:
        """Return the number of items in the dataset"""
        return len(self.pet_descriptions)
    
    def __getitem__(self, idx: int) -> Tuple[Dict, torch.Tensor]:
        """Get a pet description and its corresponding image"""
        pet = self.pet_descriptions[idx]
        
        # Load and transform image
        image_path = self.image_paths.get(pet['name'])
        if image_path:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            # Return a zero tensor if image is missing
            image = torch.zeros((3, self.image_size, self.image_size))
        
        return pet, image

def create_data_loaders(
    text_data_path: str,
    image_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 512
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        text_data_path: Path to the JSON file containing pet descriptions
        image_dir: Base directory containing pet images
        batch_size: Batch size for the data loaders
        num_workers: Number of worker processes for data loading
        image_size: Target size for images
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = RoyalPetDataset(
        text_data_path=text_data_path,
        image_dir=image_dir,
        image_size=image_size
    )
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def validate_image(image_path: str, min_size: int = 512) -> Tuple[bool, str]:
    """
    Validate an image meets our requirements
    
    Args:
        image_path: Path to the image file
        min_size: Minimum required image size
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with Image.open(image_path) as img:
            # Check format
            if img.format not in ['JPEG', 'PNG']:
                return False, f"Invalid format: {img.format}. Must be JPEG or PNG."
            
            # Check size
            if img.size[0] < min_size or img.size[1] < min_size:
                return False, f"Image too small: {img.size}. Minimum size is {min_size}x{min_size}."
            
            # Check mode
            if img.mode != 'RGB':
                return False, f"Invalid color mode: {img.mode}. Must be RGB."
            
            return True, "Image is valid"
            
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def validate_dataset(text_data_path: str, image_dir: str) -> List[str]:
    """
    Validate the entire dataset
    
    Args:
        text_data_path: Path to the JSON file containing pet descriptions
        image_dir: Base directory containing pet images
    
    Returns:
        List of error messages
    """
    errors = []
    
    # Validate text data
    try:
        with open(text_data_path, 'r') as f:
            pet_descriptions = json.load(f)
    except Exception as e:
        errors.append(f"Error loading text data: {str(e)}")
        return errors
    
    # Validate images
    image_dir = Path(image_dir)
    for species_dir in image_dir.iterdir():
        if not species_dir.is_dir():
            continue
            
        for image_path in species_dir.glob('*.jpg'):
            is_valid, error_msg = validate_image(str(image_path))
            if not is_valid:
                errors.append(f"{image_path}: {error_msg}")
    
    return errors

if __name__ == "__main__":
    # Example usage
    text_data_path = "data/text/processed_pets.json"
    image_dir = "data/images"
    
    # Validate dataset
    errors = validate_dataset(text_data_path, image_dir)
    if errors:
        print("Dataset validation errors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Dataset validation successful!")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            text_data_path=text_data_path,
            image_dir=image_dir
        )
        
        print(f"Created data loaders with {len(train_loader.dataset)} training samples "
              f"and {len(val_loader.dataset)} validation samples") 