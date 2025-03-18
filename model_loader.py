#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to load the Qwen model with different approaches
"""

import os
import torch
from transformers import AutoTokenizer, PreTrainedModel, AutoConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "Qwen/Qwen2.5-0.5B"

def attempt_load_model():
    """Try different approaches to load the model"""
    
    # Method 1: Try loading with torch_dtype specified and without TensorFlow
    logger.info("Attempt 1: Loading with torch_dtype specified and without TensorFlow")
    try:
        # Set an environment variable to disable TensorFlow
        os.environ["TRANSFORMERS_NO_TF"] = "1"
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        # Load model with specific dtype
        logger.info("Loading model with torch.float16...")
        model = PreTrainedModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        logger.info("Successfully loaded the model with Attempt 1")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Attempt 1 failed: {e}")
    
    # Method 2: Try loading via configuration
    logger.info("\nAttempt 2: Loading via configuration")
    try:
        # Load config first
        logger.info("Loading configuration...")
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        # Load model from config
        logger.info("Loading model from config...")
        model_class = config.auto_map["AutoModelForCausalLM"].split(".")[-1]
        
        # Dynamic import
        logger.info(f"Dynamically importing model class: {model_class}")
        module_path = config.auto_map["AutoModelForCausalLM"].rsplit(".", 1)[0]
        module = __import__(module_path, fromlist=[model_class])
        model_cls = getattr(module, model_class)
        
        # Create model instance
        model = model_cls.from_pretrained(
            MODEL_ID,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        logger.info("Successfully loaded the model with Attempt 2")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Attempt 2 failed: {e}")
    
    # Method 3: Try loading with local_files_only=True
    logger.info("\nAttempt 3: Loading with local_files_only=False and force_download=True")
    try:
        from transformers import AutoModelForCausalLM
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            local_files_only=False,
            force_download=True
        )
        
        # Load model with specific configuration
        logger.info("Loading model with force_download...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=False,
            force_download=True,
            device_map="auto"
        )
        
        logger.info("Successfully loaded the model with Attempt 3")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Attempt 3 failed: {e}")
    
    # Method 4: Try loading a different architecture (last resort)
    logger.info("\nAttempt 4: Loading a different model architecture")
    try:
        from transformers import AutoModelForCausalLM
        
        # Try with a different model architecture
        alternative_model = "facebook/opt-125m"
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for alternative model: {alternative_model}...")
        tokenizer = AutoTokenizer.from_pretrained(alternative_model)
        
        # Load model
        logger.info(f"Loading alternative model: {alternative_model}...")
        model = AutoModelForCausalLM.from_pretrained(
            alternative_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("Successfully loaded alternative model")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Attempt 4 failed: {e}")
    
    raise RuntimeError("All attempts to load the model failed")

if __name__ == "__main__":
    try:
        model, tokenizer = attempt_load_model()
        
        # Test the model with a simple generation
        input_text = "Write a Python function to find the maximum value in a list."
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated text: {generated_text}")
        
    except Exception as e:
        logger.error(f"Failed to load or test the model: {e}") 