#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantization script for Qwen2.5-Instruct-14B model
Supports multiple quantization methods including GPTQ, AWQ, and bitsandbytes
"""

import os
import argparse
import torch
import time
import numpy as np
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from optimum.gptq import GPTQQuantizer, load_quantized_model
from peft import prepare_model_for_kbit_training
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize Qwen2.5-Instruct-14B model")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Model identifier from Hugging Face Hub",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./quantized_models",
        help="Directory to save the quantized model"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["gptq", "awq", "bitsandbytes", "int8", "fp16"],
        default="gptq",
        help="Quantization method to use"
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8, 16],
        default=4,
        help="Number of bits for quantization"
    )
    parser.add_argument(
        "--dataset_size", 
        type=int, 
        default=1000,
        help="Size of calibration dataset for GPTQ"
    )
    parser.add_argument(
        "--block_size", 
        type=int, 
        default=2048,
        help="Maximum context size for the model"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="Quantization scheme (symmetric or asymmetric/affine)"
    )
    parser.add_argument(
        "--use_exllama",
        action="store_true",
        help="Whether to use ExLlama kernel for faster inference with GPTQ"
    )
    
    return parser.parse_args()

def get_calibration_dataset(tokenizer, dataset_size=1000, block_size=2048):
    """Create a simple calibration dataset for GPTQ quantization"""
    from datasets import load_dataset
    
    logger.info(f"Preparing calibration dataset (size: {dataset_size}, block_size: {block_size})")
    
    # Load CodeAlpaca dataset which contains programming examples
    # This should be good for coding tasks like HumanEval
    try:
        dataset = load_dataset("sahil2801/CodeAlpaca", split="train")
        dataset = dataset.select(range(min(len(dataset), dataset_size)))
    except Exception as e:
        logger.warning(f"Failed to load CodeAlpaca dataset: {e}")
        logger.info("Falling back to wikitext dataset")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        dataset = dataset.select(range(min(len(dataset), dataset_size)))
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text" if "text" in examples else "instruction"])
        return tokenized
        
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )
    
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        
        return result

    processed_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=4,
    )
    
    return processed_dataset

def print_model_size(model):
    """Print the model size in different formats"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    size_gb = size_mb / 1024
    
    logger.info(f"Model size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    logger.info(f"Number of parameters: {sum(p.nelement() for p in model.parameters()) / 1e9:.2f} billion")

def measure_inference_speed(model, tokenizer, device, num_samples=10, input_text="Write a Python function to calculate the Fibonacci sequence"):
    """Measure inference speed of the model"""
    logger.info("Measuring inference speed...")
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(**inputs, max_new_tokens=20)
    
    # Measure generation time
    start_time = time.time()
    tokens_generated = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            output = model.generate(**inputs, max_new_tokens=100, do_sample=True)
            tokens_generated += output.shape[1] - inputs.input_ids.shape[1]
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_samples
    tokens_per_sec = tokens_generated / (end_time - start_time)
    
    logger.info(f"Average generation time: {avg_time:.4f} seconds")
    logger.info(f"Tokens per second: {tokens_per_sec:.2f}")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
        logger.info(f"GPU memory reserved: {memory_reserved:.2f} GB")
    
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024**3
    logger.info(f"CPU memory usage: {cpu_memory:.2f} GB")
    
    return tokens_per_sec

def quantize_with_gptq(model, tokenizer, calibration_dataset, bits=4, use_exllama=False, device="cuda"):
    """Quantize model using GPTQ"""
    logger.info(f"Quantizing model using GPTQ with {bits} bits precision")
    
    quantizer = GPTQQuantizer(
        bits=bits,
        dataset=calibration_dataset, 
        model_seqlen=model.config.max_position_embeddings,
        use_exllama=use_exllama
    )
    
    # Quantize the model
    quantized_model = quantizer.quantize_model(model, tokenizer)
    
    return quantized_model

def quantize_with_bitsandbytes(model_id, bits=4, device="cuda"):
    """Load model directly with bitsandbytes quantization"""
    logger.info(f"Loading model with bitsandbytes quantization ({bits} bits)")
    
    # Configure quantization
    if bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # Normalized float 4
        )
    elif bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"Unsupported bits value for bitsandbytes: {bits}")
    
    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
    )
    
    # Prepare the model for potential fine-tuning
    model = prepare_model_for_kbit_training(model)
    
    return model

def quantize_with_int8(model_id, device="cuda"):
    """Load model with native int8 quantization"""
    logger.info("Loading model with int8 quantization")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.int8,
    )
    
    return model

def quantize_with_fp16(model_id, device="cuda"):
    """Load model with float16 precision"""
    logger.info("Loading model with float16 precision")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    return model

def save_quantized_model(model, tokenizer, output_dir, method, bits):
    """Save the quantized model"""
    save_path = os.path.join(output_dir, f"qwen2.5-instruct-14b-{method}-{bits}bit")
    os.makedirs(save_path, exist_ok=True)
    
    logger.info(f"Saving quantized model to {save_path}")
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    return save_path

def main():
    args = parse_args()
    
    logger.info(f"Starting quantization of {args.model_id} using {args.method} method with {args.bits} bits")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare calibration dataset for GPTQ
    if args.method == "gptq":
        calibration_dataset = get_calibration_dataset(
            tokenizer, 
            dataset_size=args.dataset_size, 
            block_size=args.block_size
        )
    
    # Load and quantize model based on selected method
    start_time = time.time()
    
    if args.method == "gptq":
        # First load the base model in FP16 to save memory
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        
        # Print original model size
        logger.info("Original model details:")
        print_model_size(model)
        
        # Quantize using GPTQ
        quantized_model = quantize_with_gptq(
            model, 
            tokenizer, 
            calibration_dataset, 
            bits=args.bits,
            use_exllama=args.use_exllama,
            device=args.device
        )
        
    elif args.method == "bitsandbytes":
        quantized_model = quantize_with_bitsandbytes(
            args.model_id, 
            bits=args.bits, 
            device=args.device
        )
        
    elif args.method == "int8":
        quantized_model = quantize_with_int8(
            args.model_id, 
            device=args.device
        )
        
    elif args.method == "fp16":
        quantized_model = quantize_with_fp16(
            args.model_id, 
            device=args.device
        )
        
    else:
        raise ValueError(f"Unsupported quantization method: {args.method}")
    
    quantization_time = time.time() - start_time
    logger.info(f"Quantization completed in {quantization_time:.2f} seconds")
    
    # Print quantized model details
    logger.info("Quantized model details:")
    print_model_size(quantized_model)
    
    # Measure inference speed
    tokens_per_sec = measure_inference_speed(quantized_model, tokenizer, args.device)
    
    # Save the quantized model
    save_path = save_quantized_model(
        quantized_model, 
        tokenizer, 
        args.output_dir, 
        args.method, 
        args.bits
    )
    
    logger.info(f"Model successfully quantized and saved to {save_path}")
    logger.info(f"Method: {args.method}, Bits: {args.bits}, Scheme: {args.scheme}")
    logger.info(f"Inference speed: {tokens_per_sec:.2f} tokens/sec")

if __name__ == "__main__":
    main() 