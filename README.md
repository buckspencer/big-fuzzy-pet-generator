# Big Fuzzy Pet Generator

A Replicate model that generates detailed descriptions and images of royal pets. This model creates personalized pet portraits with matching backstories suitable for physical production (printing on canvas, laser etching on wood/metal).

## Features

- Generates detailed pet descriptions in structured JSON format
- Creates high-quality pet images matching the descriptions
- Focuses on royal and dignified pet characteristics
- Supports multiple pet types (cats, dogs, birds, etc.)
- Produces consistent, parseable output

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
├── cog.yaml             # Replicate/Cog configuration
├── predict.py           # Main model implementation
└── requirements.txt     # Python dependencies
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

### Images
- Format: JPEG or PNG
- Resolution: Minimum 512x512 pixels
- Quality: High quality, well-lit, clear focus
- Style: Professional photography or high-quality digital art
- Content: Single pet per image, centered composition
- Background: Clean, simple backgrounds preferred

### Text Descriptions
- Follow the template structure in `data/text/template.json`
- Include detailed physical descriptions
- Provide realistic abilities and achievements
- Maintain royal/dignified theme

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 