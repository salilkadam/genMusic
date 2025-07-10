#!/usr/bin/env python3
"""
Model download script for Music Generation app.
This script downloads and caches the required models to avoid startup delays.
"""

import os
import sys
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, MusicgenForConditionalGeneration, AutoProcessor
import torch

# Set cache directory
CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE', '/models')
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR

def download_musicgen_model():
    """Download the MusicGen Large model."""
    print("Downloading MusicGen Large model...")
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Download model using pipeline (this will cache it)
        model_id = "facebook/musicgen-large"
        print(f"Downloading model: {model_id}")
        
        # Download model and processor separately to force safetensors
        print("Downloading model files...")
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            use_safetensors=True,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR
        )
        
        print("Model and processor downloaded successfully!")
        
        # Move to device
        model = model.to(device)
        
        # Create pipeline for compatibility
        synthesiser = pipeline(
            "text-to-audio",
            model=model,
            tokenizer=processor,
            device=0 if device.type == 'cuda' else -1
        )
        
        print(f"Model downloaded successfully and cached to: {CACHE_DIR}")
        
        # Clean up
        del synthesiser
        del model
        del processor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def main():
    """Main function to download all required models."""
    print("Starting model download process...")
    print(f"Cache directory: {CACHE_DIR}")
    
    # Create cache directory if it doesn't exist
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Download MusicGen model
    success = download_musicgen_model()
    
    if success:
        print("\nAll models downloaded successfully!")
        print("You can now start the application without model download delays.")
    else:
        print("\nFailed to download some models.")
        sys.exit(1)

if __name__ == "__main__":
    main() 