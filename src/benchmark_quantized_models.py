#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmarking script for quantized LLMs
Evaluates model performance metrics including:
- Inference speed
- Memory usage 
- Model size
- HumanEval benchmark performance
"""

import os
import argparse
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import psutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define sample prompts for inference speed testing
SAMPLE_PROMPTS = [
    "Write a Python function to find the maximum value in a binary tree.",
    "Create a JavaScript function to calculate the Fibonacci sequence recursively.",
    "Implement a sorting algorithm in Python that has O(n log n) time complexity.",
    "Write a function to check if a string is a palindrome in C++.",
    "Create a function to find all prime numbers up to n using the Sieve of Eratosthenes."
]

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark quantized models")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the quantized model or model ID from Hugging Face Hub",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--run_humaneval",
        action="store_true",
        help="Whether to run HumanEval benchmark (requires bigcode-evaluation-harness)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to generate for measuring inference speed"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Whether to save benchmark results to disk"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to create visualizations of benchmark results"
    )
    
    return parser.parse_args()

def load_model(model_path, device="cuda"):
    """Load model from local path or Hugging Face Hub"""
    logger.info(f"Loading model from {model_path}")
    
    start_time = time.time()
    
    try:
        # Try loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Try loading model with auto-detection of quantization format
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    return model, tokenizer, load_time

def measure_model_size(model):
    """Measure model size in memory and number of parameters"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    size_gb = size_mb / 1024
    
    # Count parameters
    param_count = sum(p.nelement() for p in model.parameters())
    param_count_billions = param_count / 1e9
    
    logger.info(f"Model size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    logger.info(f"Number of parameters: {param_count_billions:.2f} billion")
    
    return {
        "size_mb": size_mb,
        "size_gb": size_gb,
        "param_count": param_count,
        "param_count_billions": param_count_billions
    }

def measure_memory_usage(model, device="cuda"):
    """Measure memory usage of the model"""
    memory_stats = {}
    
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024**3  # GB
    memory_stats["cpu_memory_gb"] = cpu_memory
    
    # GPU memory (if available)
    if device == "cuda" and torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_stats["gpu_memory_allocated_gb"] = gpu_memory_allocated
        memory_stats["gpu_memory_reserved_gb"] = gpu_memory_reserved
    
    logger.info(f"CPU memory usage: {cpu_memory:.2f} GB")
    if device == "cuda" and torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {memory_stats['gpu_memory_allocated_gb']:.2f} GB")
        logger.info(f"GPU memory reserved: {memory_stats['gpu_memory_reserved_gb']:.2f} GB")
    
    return memory_stats

def measure_inference_speed(model, tokenizer, device="cuda", n_samples=5):
    """Measure inference speed across different sample prompts"""
    logger.info("Measuring inference speed...")
    
    results = []
    
    # Run inference on each prompt
    for prompt in SAMPLE_PROMPTS:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = model.generate(**inputs, max_new_tokens=20)
        
        # Measure generation time
        start_time = time.time()
        tokens_generated = 0
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = model.generate(**inputs, max_new_tokens=100, do_sample=True)
                tokens_generated += output.shape[1] - inputs.input_ids.shape[1]
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_samples
        tokens_per_sec = tokens_generated / (end_time - start_time)
        
        results.append({
            "prompt": prompt,
            "avg_generation_time": avg_time,
            "tokens_per_second": tokens_per_sec
        })
        
        logger.info(f"Prompt: {prompt[:30]}...")
        logger.info(f"Average generation time: {avg_time:.4f} seconds")
        logger.info(f"Tokens per second: {tokens_per_sec:.2f}")
    
    # Calculate overall averages
    avg_generation_time = np.mean([r["avg_generation_time"] for r in results])
    avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in results])
    
    logger.info(f"Overall average generation time: {avg_generation_time:.4f} seconds")
    logger.info(f"Overall average tokens per second: {avg_tokens_per_second:.2f}")
    
    return {
        "detailed_results": results,
        "avg_generation_time": avg_generation_time,
        "avg_tokens_per_second": avg_tokens_per_second
    }

def run_humaneval_benchmark(model_path):
    """Run HumanEval benchmark using bigcode-evaluation-harness"""
    logger.info("Running HumanEval benchmark...")
    
    # Check if we're in the bigcode-evaluation-harness directory
    if not os.path.exists("main.py") and os.path.exists("bigcode-evaluation-harness"):
        os.chdir("bigcode-evaluation-harness")
    
    # Create a configuration file for accelerate
    with open("config.yaml", "w") as f:
        f.write("""compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false""")
    
    # Run HumanEval with the model
    command = f"accelerate launch --config_file config.yaml main.py \
                --model {model_path} \
                --max_length_generation 512 \
                --tasks humaneval \
                --temperature 0.2 \
                --limit 50 \
                --n_samples 20 \
                --batch_size 10 \
                --allow_code_execution"
    
    logger.info(f"Running command: {command}")
    result = os.system(command)
    
    if result != 0:
        logger.error("HumanEval benchmark failed to run")
        return None
    
    # Get the most recent results file
    results_dir = Path("results")
    if not results_dir.exists():
        logger.error("Results directory not found")
        return None
    
    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        logger.error("No result files found")
        return None
    
    latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
    
    # Load the results
    with open(latest_result, "r") as f:
        humaneval_results = json.load(f)
    
    logger.info(f"HumanEval results loaded from {latest_result}")
    
    # Extract key metrics
    pass_at_1 = humaneval_results.get("pass@1", None)
    if pass_at_1 is not None:
        logger.info(f"HumanEval Pass@1: {pass_at_1:.4f}")
    
    return humaneval_results

def visualize_results(benchmark_results, output_dir="./benchmark_results"):
    """Create visualizations of benchmark results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create dataframe for easier plotting
    model_name = benchmark_results["model_name"]
    
    # Bar chart for tokens per second
    plt.figure(figsize=(10, 6))
    plt.bar(model_name, benchmark_results["inference_speed"]["avg_tokens_per_second"], color="blue")
    plt.title("Inference Speed (Tokens per Second)")
    plt.ylabel("Tokens/sec")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_speed.png"))
    
    # Memory usage
    if "gpu_memory_allocated_gb" in benchmark_results["memory_usage"]:
        memory_types = ["CPU Memory", "GPU Allocated", "GPU Reserved"]
        memory_values = [
            benchmark_results["memory_usage"]["cpu_memory_gb"],
            benchmark_results["memory_usage"]["gpu_memory_allocated_gb"],
            benchmark_results["memory_usage"]["gpu_memory_reserved_gb"]
        ]
    else:
        memory_types = ["CPU Memory"]
        memory_values = [benchmark_results["memory_usage"]["cpu_memory_gb"]]
    
    plt.figure(figsize=(10, 6))
    plt.bar(memory_types, memory_values, color="green")
    plt.title(f"Memory Usage - {model_name}")
    plt.ylabel("Memory (GB)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_usage.png"))
    
    # Model size visualization
    plt.figure(figsize=(8, 8))
    plt.pie([benchmark_results["model_size"]["size_gb"]], 
            labels=[f"{benchmark_results['model_size']['size_gb']:.2f} GB"],
            autopct='%1.1f%%', 
            startangle=90)
    plt.title(f"Model Size - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_size.png"))
    
    # Human eval results if available
    if benchmark_results.get("humaneval_results"):
        if "pass@1" in benchmark_results["humaneval_results"]:
            plt.figure(figsize=(10, 6))
            plt.bar(model_name, benchmark_results["humaneval_results"]["pass@1"], color="orange")
            plt.title("HumanEval Pass@1 Score")
            plt.ylabel("Pass@1")
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "humaneval_results.png"))
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Extract model name from path
    model_name = os.path.basename(args.model_path.rstrip("/"))
    
    # Start benchmarking
    benchmark_results = {
        "model_name": model_name,
        "model_path": args.model_path,
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Load model
    model, tokenizer, load_time = load_model(args.model_path, args.device)
    benchmark_results["load_time"] = load_time
    
    # Measure model size
    benchmark_results["model_size"] = measure_model_size(model)
    
    # Measure memory usage
    benchmark_results["memory_usage"] = measure_memory_usage(model, args.device)
    
    # Measure inference speed
    benchmark_results["inference_speed"] = measure_inference_speed(
        model, tokenizer, args.device, args.n_samples
    )
    
    # Run HumanEval benchmark if requested
    if args.run_humaneval:
        benchmark_results["humaneval_results"] = run_humaneval_benchmark(args.model_path)
    
    # Save results
    if args.save_results:
        results_file = os.path.join(args.output_dir, f"{model_name}_benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Benchmark results saved to {results_file}")
    
    # Create visualizations
    if args.visualize:
        visualize_results(benchmark_results, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print(f"BENCHMARK SUMMARY FOR {model_name}")
    print("="*50)
    print(f"Model size: {benchmark_results['model_size']['size_gb']:.2f} GB")
    print(f"Parameters: {benchmark_results['model_size']['param_count_billions']:.2f} billion")
    print(f"Inference speed: {benchmark_results['inference_speed']['avg_tokens_per_second']:.2f} tokens/sec")
    print(f"Avg. generation time: {benchmark_results['inference_speed']['avg_generation_time']:.4f} seconds")
    
    if "gpu_memory_allocated_gb" in benchmark_results["memory_usage"]:
        print(f"GPU memory allocated: {benchmark_results['memory_usage']['gpu_memory_allocated_gb']:.2f} GB")
    print(f"CPU memory usage: {benchmark_results['memory_usage']['cpu_memory_gb']:.2f} GB")
    
    if args.run_humaneval and benchmark_results.get("humaneval_results"):
        if "pass@1" in benchmark_results["humaneval_results"]:
            print(f"HumanEval Pass@1: {benchmark_results['humaneval_results']['pass@1']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main() 