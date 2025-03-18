#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal PyTorch-only script to test model loading without TensorFlow dependencies
"""

import os
import time
import torch
import numpy as np
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Explicitly disable TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable TensorFlow logging

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test PyTorch model loading")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="facebook/opt-125m",
        help="Model ID to load (use small models)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./test_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    
    return parser.parse_args()

def load_model_pytorch_only(model_id, device="cpu"):
    """Load model with PyTorch only, avoiding TensorFlow completely"""
    logger.info(f"Loading model {model_id} using PyTorch only")
    
    # Import transformers here to ensure TRANSFORMERS_NO_TF is set
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    start_time = time.time()
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        return True, model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False, None, None

def test_model_generation(model, tokenizer, device="cpu"):
    """Test model generation"""
    logger.info("Testing model generation...")
    
    prompt = "Write a simple Python function to calculate the factorial of a number:"
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to(device)
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, do_sample=True)
        
        # Decode output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generation_time = time.time() - start_time
        
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        logger.info(f"Generated text: {generated_text}")
        
        return True
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return False

def test_alternative_model(model_id="gpt2", device="cpu"):
    """Try with a different model if the first one fails"""
    logger.info(f"Trying alternative model: {model_id}")
    
    # Import transformers here to ensure TRANSFORMERS_NO_TF is set
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
        )
        
        logger.info("Model loaded successfully")
        
        # Test generation
        test_model_generation(model, tokenizer, device)
        
        return True
    except Exception as e:
        logger.error(f"Error loading alternative model: {e}")
        return False

def main():
    """Main function"""
    args = parse_args()
    
    logger.info("Starting PyTorch-only model test")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Try loading the model
    success, model, tokenizer = load_model_pytorch_only(args.model_id, args.device)
    
    if success:
        # Test model generation
        gen_success = test_model_generation(model, tokenizer, args.device)
        if not gen_success:
            logger.warning("Generation test failed, trying alternative model")
            test_alternative_model("gpt2", args.device)
    else:
        # If loading fails, try a different model
        logger.warning("Model loading failed, trying alternative model")
        test_alternative_model("gpt2", args.device)
    
    logger.info("Test completed")

if __name__ == "__main__":
    main() 