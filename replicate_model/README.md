# Big Fuzzy Pet Generator

A Replicate model that generates detailed descriptions and images of royal pets. This model creates personalized pet portraits with matching backstories suitable for physical production (printing on canvas, laser etching on wood/metal).

## Features

- Generates detailed pet descriptions in structured JSON format
- Creates high-quality pet images matching the descriptions
- Focuses on royal and dignified pet characteristics
- Supports multiple pet types (cats, dogs, birds, etc.)
- Produces consistent, parseable output
- Supports incremental training for continuous improvement
- Includes utilities for image management and validation

## Project Structure

```
big-fuzzy-pet-generator/
├── data/
│   ├── images/           # Training images organized by species
│   │   ├── cats/
│   │   ├── dogs/
│   │   ├── birds/
│   │   └── other/
│   └── text/            # Pet descriptions and templates
├── models/              # Trained model checkpoints
│   ├── text_model/     # Text generation model
│   └── image_model/    # Image generation model
├── utils/              # Utility scripts
│   └── rename_images.py # Image renaming and validation
├── cog.yaml            # Replicate/Cog configuration
├── predict.py          # Main model implementation
├── train.py           # Training script
└── requirements.txt    # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Add pet descriptions following the template in `data/text/template.json`
   - Add corresponding images to the appropriate species directories
   - Run validation:
   ```bash
   python data/data_loader.py
   ```

## Image Management

### Renaming Images
When adding new images, use the utility script to automatically rename them to match our required format:

```bash
python utils/rename_images.py
```

This script will:
1. Process each species directory in `data/images/`
2. Validate images (format, size, etc.)
3. Rename images to format: `species_XXX.jpg`
4. Create backups of original filenames
5. Skip invalid images with warnings

Example:
```bash
Processing cats directory...
Enter starting number for cats images (default: 1): 1
Renamed: IMG_1234.jpg -> cats_001.jpg
Renamed: photo.jpg -> cats_002.jpg
```

### Image Requirements
- Format: JPEG or PNG
- Resolution: Minimum 512x512 pixels
- Quality: High quality, well-lit, clear focus
- Style: Professional photography or high-quality digital art
- Content: Single pet per image, centered composition
- Background: Clean, simple backgrounds preferred
- Naming: `species_XXX.jpg` (e.g., `cats_001.jpg`)

## Training the Model

### Initial Training
To train the model for the first time with your initial dataset:

```bash
python train.py
```

This will:
- Train the text model (TinyLlama) on your descriptions
- Train the image model (Stable Diffusion XL) on your images
- Save checkpoints in the `models/` directory

### Incremental Training
You can continuously improve the model by adding more data and retraining:

1. Add new data:
   - Place new images in the appropriate species folders
   - Run the image renaming utility:
   ```bash
   python utils/rename_images.py
   ```
   - Add new descriptions to your JSON files

2. Retrain the model:
   ```bash
   python train.py
   ```
   The script will automatically:
   - Find the latest checkpoint
   - Resume training from that point
   - Save new checkpoints

3. Monitor training progress:
   ```bash
   # View text model training progress
   tensorboard --logdir models/text_model/logs
   
   # View image model training progress
   tensorboard --logdir models/image_model/logs
   ```

### Training Parameters
You can adjust training parameters in `train.py`:

```python
# Text model parameters
train_text_model(
    num_train_epochs=3,
    batch_size=4,
    learning_rate=2e-5
)

# Image model parameters
train_image_model(
    num_train_epochs=3,
    batch_size=1,
    learning_rate=1e-6
)
```

## Usage

The model can be used through Replicate's API:

```python
import replicate

output = replicate.run(
    "buckspencer/big-fuzzy-pet-generator",
    input={
        "prompt": "A dignified British Shorthair cat with royal bearing"
    }
)
```

## Training Data Requirements

### Text Descriptions
- Follow the template structure in `data/text/template.json`
- Include detailed physical descriptions
- Provide realistic abilities and achievements
- Maintain royal/dignified theme

## Model Checkpoints

The training process saves checkpoints in the `models/` directory:
- Keeps the last 3 checkpoints to save space
- Automatically resumes from the latest checkpoint
- Each training run is timestamped for tracking

## Deployment

After training or updating the model:

1. Prepare for Replicate:
```bash
python prepare_replicate.py
```

2. Deploy to Replicate:
```bash
cd replicate_model
cog push
```

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 