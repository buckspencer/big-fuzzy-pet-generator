import os
import shutil
from pathlib import Path

def prepare_replicate():
    """Prepare the model for Replicate deployment"""
    # Create replicate_model directory
    replicate_dir = Path("replicate_model")
    replicate_dir.mkdir(exist_ok=True)
    
    # Copy necessary files
    files_to_copy = [
        "predict.py",
        "cog.yaml",
        "requirements.txt",
        "README.md",
        "LICENSE"
    ]
    
    for file in files_to_copy:
        if Path(file).exists():
            shutil.copy2(file, replicate_dir)
    
    # Copy model checkpoints
    models_dir = replicate_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    if Path("models").exists():
        for model_type in ["text_model", "image_model"]:
            src = Path("models") / model_type
            if src.exists():
                dst = models_dir / model_type
                shutil.copytree(src, dst, dirs_exist_ok=True)
    
    # Create .gitignore
    with open(replicate_dir / ".gitignore", "w") as f:
        f.write("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Model files
models/*.bin
models/*.pt
models/*.pth
models/*.ckpt
""")
    
    print("Model prepared for Replicate deployment!")
    print("\nNext steps:")
    print("1. cd replicate_model")
    print("2. cog push")
    print("3. Follow the prompts to deploy to Replicate")

if __name__ == "__main__":
    prepare_replicate() 