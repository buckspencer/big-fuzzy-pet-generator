import os
from pathlib import Path
import shutil
from PIL import Image
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_image_info(image_path: Path) -> Optional[tuple]:
    """
    Get image information including format and dimensions
    Returns: (width, height, format) or None if invalid
    """
    try:
        with Image.open(image_path) as img:
            return img.size[0], img.size[1], img.format
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {e}")
        return None

def validate_image(image_path: Path, min_size: int = 512) -> bool:
    """Validate image meets our requirements"""
    info = get_image_info(image_path)
    if not info:
        return False
    
    width, height, format = info
    
    # Check format
    if format not in ['JPEG', 'PNG']:
        logger.warning(f"Invalid format for {image_path}: {format}. Must be JPEG or PNG.")
        return False
    
    # Check size
    if width < min_size or height < min_size:
        logger.warning(f"Image too small: {image_path} ({width}x{height}). Minimum size is {min_size}x{min_size}.")
        return False
    
    return True

def rename_images(image_dir: str, species: str, start_number: int = 1):
    """
    Rename images in a species directory to match our format
    Format: species_XXX.jpg
    """
    image_dir = Path(image_dir)
    species = species.lower()
    
    if not image_dir.exists():
        logger.error(f"Directory not found: {image_dir}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
    
    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return
    
    # Create backup directory
    backup_dir = image_dir / 'original_names'
    backup_dir.mkdir(exist_ok=True)
    
    # Rename images
    for idx, image_path in enumerate(image_files, start=start_number):
        # Validate image
        if not validate_image(image_path):
            continue
        
        # Create new filename
        new_name = f"{species}_{idx:03d}.jpg"
        new_path = image_dir / new_name
        
        # Backup original
        backup_path = backup_dir / image_path.name
        shutil.copy2(image_path, backup_path)
        
        # Rename file
        try:
            image_path.rename(new_path)
            logger.info(f"Renamed: {image_path.name} -> {new_name}")
        except Exception as e:
            logger.error(f"Error renaming {image_path}: {e}")

def process_species_directories(base_dir: str):
    """Process all species directories in the base directory"""
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        return
    
    # Process each species directory
    for species_dir in base_dir.iterdir():
        if not species_dir.is_dir():
            continue
        
        species = species_dir.name
        logger.info(f"Processing {species} directory...")
        
        # Get starting number
        try:
            start_number = int(input(f"Enter starting number for {species} images (default: 1): ").strip() or "1")
        except ValueError:
            start_number = 1
        
        rename_images(species_dir, species, start_number)

if __name__ == "__main__":
    # Process all species directories
    process_species_directories("data/images") 