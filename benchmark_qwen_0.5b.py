#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark Script for Quantized Qwen 0.5B Model

This script:
1. Loads the Qwen 0.5B model
2. Quantizes it using various methods
3. Benchmarks the quantized model with HumanEval
4. Provides visualizations of performance metrics
"""

import os
import argparse
import time
import json
import torch
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import logging
import subprocess
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "Qwen/Qwen2.5-0.5B"
SAMPLE_PROMPTS = [
    "Write a Python function to find the maximum value in a binary tree.",
    "Create a JavaScript function to calculate the Fibonacci sequence recursively.",
    "Implement a sorting algorithm in Python that has O(n log n) time complexity.",
    "Write a function to check if a string is a palindrome in C++.",
    "Create a function to find all prime numbers up to n using the Sieve of Eratosthenes."
]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark Quantized Qwen 0.5B Model")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./qwen_0.5b_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="fp16,int8,int4",
        help="Comma-separated list of quantization methods to benchmark"
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
        "--run_humaneval",
        action="store_true",
        help="Whether to run HumanEval benchmark (requires bigcode-evaluation-harness)"
    )
    parser.add_argument(
        "--humaneval_limit",
        type=int,
        default=20,
        help="Number of HumanEval problems to evaluate"
    )
    
    return parser.parse_args()

def setup_environment(args):
    """Setup environment and directories"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for required packages
    try:
        import plotly
    except ImportError:
        logger.info("Installing plotly...")
        subprocess.run(["pip", "install", "plotly"], check=True)
    
    # Set up HumanEval benchmark if needed
    if args.run_humaneval:
        if not os.path.exists("bigcode-evaluation-harness"):
            logger.info("Cloning bigcode-evaluation-harness repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/bigcode-project/bigcode-evaluation-harness.git"],
                check=True
            )
            logger.info("Installing bigcode-evaluation-harness requirements...")
            subprocess.run(
                ["pip", "install", "-e", "./bigcode-evaluation-harness"],
                check=True
            )

def load_base_model(device="cuda"):
    """Load the base Qwen 0.5B model"""
    logger.info(f"Loading base model {MODEL_ID}")
    
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    load_time = time.time() - start_time
    
    logger.info(f"Base model loaded in {load_time:.2f} seconds")
    return model, tokenizer, load_time

def quantize_model(model_id, method="fp16", device="cuda"):
    """Quantize the model using the specified method"""
    logger.info(f"Quantizing model using {method}")
    
    start_time = time.time()
    
    # First load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if method == "fp16":
        # Load model in fp16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    elif method == "int8":
        # Load model with 8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    elif method == "int4":
        # Load model with 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # Normalized float 4
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    else:
        raise ValueError(f"Unsupported quantization method: {method}")
    
    quantize_time = time.time() - start_time
    logger.info(f"Model quantized in {quantize_time:.2f} seconds")
    
    return model, tokenizer, quantize_time

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
    logger.info(f"Number of parameters: {param_count_billions:.4f} billion")
    
    return {
        "size_mb": size_mb,
        "size_gb": size_gb,
        "param_count": param_count,
        "param_count_billions": param_count_billions
    }

def measure_memory_usage(device="cuda"):
    """Measure memory usage"""
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

def run_humaneval_benchmark(model_path, limit=20):
    """Run HumanEval benchmark using bigcode-evaluation-harness"""
    logger.info("Running HumanEval benchmark...")
    
    humaneval_dir = "bigcode-evaluation-harness"
    if not os.path.exists(humaneval_dir):
        raise FileNotFoundError(
            "bigcode-evaluation-harness directory not found. "
            "Run with --setup_humaneval to set it up."
        )
    
    # Change to the evaluation harness directory
    original_dir = os.getcwd()
    os.chdir(humaneval_dir)
    
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
                --limit {limit} \
                --n_samples 20 \
                --batch_size 10 \
                --allow_code_execution"
    
    logger.info(f"Running command: {command}")
    result = os.system(command)
    
    # Change back to original directory
    os.chdir(original_dir)
    
    if result != 0:
        logger.error("HumanEval benchmark failed to run")
        return None
    
    # Find the latest result file
    results_dir = os.path.join(humaneval_dir, "results")
    if not os.path.exists(results_dir):
        logger.error("Results directory not found")
        return None
    
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    if not result_files:
        logger.error("No result files found")
        return None
    
    # Sort by creation time (newest first)
    result_files.sort(key=lambda x: os.path.getctime(os.path.join(results_dir, x)), reverse=True)
    latest_result = os.path.join(results_dir, result_files[0])
    
    # Load the results
    with open(latest_result, "r") as f:
        humaneval_results = json.load(f)
    
    logger.info(f"HumanEval results loaded from {latest_result}")
    
    # Extract Pass@1 score
    pass_at_1 = humaneval_results.get("pass@1", None)
    if pass_at_1 is not None:
        logger.info(f"HumanEval Pass@1: {pass_at_1:.4f}")
    
    return humaneval_results

def create_visualizations(benchmark_results, output_dir):
    """Create visualizations of benchmark results"""
    logger.info("Creating visualizations...")
    
    # Convert results to DataFrame for easier visualization
    methods = list(benchmark_results.keys())
    model_sizes = [r["model_size"]["size_mb"] / 1024 for r in benchmark_results.values()]  # Convert to GB
    inference_speeds = [r["inference_speed"]["avg_tokens_per_second"] for r in benchmark_results.values()]
    
    # Extract memory usage
    gpu_memory = []
    for r in benchmark_results.values():
        if "gpu_memory_allocated_gb" in r["memory_usage"]:
            gpu_memory.append(r["memory_usage"]["gpu_memory_allocated_gb"])
        else:
            gpu_memory.append(0)
    
    # Extract HumanEval results if available
    humaneval_scores = []
    for r in benchmark_results.values():
        if "humaneval_results" in r and r["humaneval_results"] and "pass@1" in r["humaneval_results"]:
            humaneval_scores.append(r["humaneval_results"]["pass@1"])
        else:
            humaneval_scores.append(None)
    
    df = pd.DataFrame({
        "Method": methods,
        "Model Size (GB)": model_sizes,
        "Inference Speed (tokens/sec)": inference_speeds,
        "GPU Memory (GB)": gpu_memory
    })
    
    if all(score is not None for score in humaneval_scores):
        df["HumanEval Pass@1"] = humaneval_scores
    
    # Save data to CSV
    df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # 1. Matplotlib: Model Size Comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Method", y="Model Size (GB)", data=df, palette="Blues_d")
    plt.title("Model Size Comparison")
    plt.ylabel("Size (GB)")
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(model_sizes):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_size_comparison.png"))
    plt.close()
    
    # 2. Matplotlib: Inference Speed Comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Method", y="Inference Speed (tokens/sec)", data=df, palette="Greens_d")
    plt.title("Inference Speed Comparison")
    plt.ylabel("Tokens per Second")
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(inference_speeds):
        ax.text(i, v + 0.5, f"{v:.2f}", ha="center")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_speed_comparison.png"))
    plt.close()
    
    # 3. Matplotlib: Memory Usage Comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Method", y="GPU Memory (GB)", data=df, palette="Reds_d")
    plt.title("GPU Memory Usage Comparison")
    plt.ylabel("Memory (GB)")
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(gpu_memory):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_usage_comparison.png"))
    plt.close()
    
    # 4. Seaborn: Combined Metrics
    if "HumanEval Pass@1" in df.columns:
        plt.figure(figsize=(12, 10))
        
        # Normalize metrics for comparison
        metrics = ["Model Size (GB)", "Inference Speed (tokens/sec)", "GPU Memory (GB)", "HumanEval Pass@1"]
        normalized_df = df.copy()
        
        for metric in metrics:
            if metric == "Model Size (GB)" or metric == "GPU Memory (GB)":
                # For these metrics, smaller is better, so invert the normalization
                max_val = normalized_df[metric].max()
                min_val = normalized_df[metric].min()
                normalized_df[f"Normalized {metric}"] = 1 - ((normalized_df[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0)
            else:
                # For these metrics, larger is better
                max_val = normalized_df[metric].max()
                min_val = normalized_df[metric].min()
                normalized_df[f"Normalized {metric}"] = (normalized_df[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0
        
        # Melt the dataframe for easier plotting
        plot_df = normalized_df.melt(
            id_vars=["Method"], 
            value_vars=[f"Normalized {m}" for m in metrics], 
            var_name="Metric", 
            value_name="Normalized Value"
        )
        
        # Create heatmap
        pivot_df = plot_df.pivot(index="Method", columns="Metric", values="Normalized Value")
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Normalized Performance Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "normalized_metrics_heatmap.png"))
        plt.close()
        
        # 5. Plotly: Combined Performance Visualization
        metrics_to_plot = ["Inference Speed (tokens/sec)", "HumanEval Pass@1"]
        fig = make_subplots(rows=1, cols=1)
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=df["Model Size (GB)"],
                y=df["Inference Speed (tokens/sec)"],
                mode='markers',
                marker=dict(
                    size=df["HumanEval Pass@1"] * 100,  # Size based on HumanEval score
                    color=df.index,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Method Index")
                ),
                text=df["Method"],
                hovertemplate=
                "<b>%{text}</b><br>" +
                "Model Size: %{x:.2f} GB<br>" +
                "Inference Speed: %{y:.2f} tokens/sec<br>" +
                "HumanEval Pass@1: %{marker.size:.2f}%",
            )
        )
        
        fig.update_layout(
            title="Performance Trade-offs Between Model Size, Speed, and Accuracy",
            xaxis_title="Model Size (GB)",
            yaxis_title="Inference Speed (tokens/sec)",
            legend_title="Method",
            width=800,
            height=600
        )
        
        fig.write_html(os.path.join(output_dir, "performance_tradeoffs.html"))
    
    logger.info(f"Visualizations saved to {output_dir}")

def benchmark_model(model, tokenizer, method, device="cuda", n_samples=5, run_humaneval=False, humaneval_limit=20):
    """Run all benchmarks on a quantized model"""
    benchmark_results = {}
    
    # Measure model size
    benchmark_results["model_size"] = measure_model_size(model)
    
    # Measure memory usage
    benchmark_results["memory_usage"] = measure_memory_usage(device)
    
    # Measure inference speed
    benchmark_results["inference_speed"] = measure_inference_speed(model, tokenizer, device, n_samples)
    
    # Run HumanEval benchmark if requested
    if run_humaneval:
        # Create a temporary directory to save the model
        model_dir = f"temp_{method}_model"
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Saving model to {model_dir} for HumanEval evaluation")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        # Run HumanEval
        benchmark_results["humaneval_results"] = run_humaneval_benchmark(model_dir, humaneval_limit)
        
        # Clean up
        # import shutil
        # shutil.rmtree(model_dir)
    
    return benchmark_results

def main():
    """Main function"""
    args = parse_args()
    
    # Setup environment
    setup_environment(args)
    
    # Parse quantization methods
    methods = args.methods.split(",")
    
    # Dictionary to store all benchmark results
    all_results = {}
    
    # Benchmark each quantization method
    for method in methods:
        logger.info(f"Starting benchmark for {method} quantization")
        
        # Quantize model
        model, tokenizer, _ = quantize_model(MODEL_ID, method, args.device)
        
        # Run benchmarks
        benchmark_result = benchmark_model(
            model, 
            tokenizer, 
            method, 
            args.device, 
            args.n_samples, 
            args.run_humaneval, 
            args.humaneval_limit
        )
        
        # Store results
        all_results[method] = benchmark_result
        
        # Free up memory
        del model, tokenizer
        torch.cuda.empty_cache() if args.device == "cuda" else None
    
    # Create visualizations
    create_visualizations(all_results, args.output_dir)
    
    # Save complete results as JSON
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        # Convert non-serializable objects to strings
        serializable_results = json.dumps(all_results, default=lambda o: str(o), indent=4)
        f.write(serializable_results)
    
    # Generate report
    generate_report(all_results, args.output_dir)
    
    logger.info(f"Benchmarking complete. Results saved to {args.output_dir}")

def generate_report(benchmark_results, output_dir):
    """Generate a comprehensive report"""
    report_path = os.path.join(output_dir, "benchmark_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Qwen 0.5B Quantization Benchmark Report\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report presents the benchmarking results for the Qwen 0.5B model under different quantization schemes.\n\n")
        
        f.write("## Model Information\n\n")
        f.write(f"- **Base Model**: {MODEL_ID}\n")
        f.write(f"- **Architecture**: Transformer-based language model\n")
        f.write(f"- **Parameter Count**: {benchmark_results[list(benchmark_results.keys())[0]]['model_size']['param_count_billions']:.4f} billion\n\n")
        
        f.write("## Quantization Methods\n\n")
        f.write("| Method | Model Size (GB) | Inference Speed (tokens/sec) | GPU Memory (GB) |\n")
        f.write("|--------|----------------|------------------------------|----------------|\n")
        
        for method, results in benchmark_results.items():
            model_size = results["model_size"]["size_gb"]
            inference_speed = results["inference_speed"]["avg_tokens_per_second"]
            gpu_memory = results["memory_usage"].get("gpu_memory_allocated_gb", "N/A")
            
            f.write(f"| {method} | {model_size:.4f} | {inference_speed:.2f} | {gpu_memory if isinstance(gpu_memory, str) else f'{gpu_memory:.2f}'} |\n")
        
        f.write("\n## HumanEval Results\n\n")
        
        # Check if HumanEval results are available
        humaneval_available = all("humaneval_results" in r and r["humaneval_results"] and "pass@1" in r["humaneval_results"] for r in benchmark_results.values())
        
        if humaneval_available:
            f.write("| Method | Pass@1 |\n")
            f.write("|--------|--------|\n")
            
            for method, results in benchmark_results.items():
                pass_at_1 = results["humaneval_results"]["pass@1"]
                f.write(f"| {method} | {pass_at_1:.4f} |\n")
        else:
            f.write("HumanEval benchmark results not available for all methods.\n")
        
        f.write("\n## Performance Analysis\n\n")
        f.write("### Key Observations\n\n")
        
        # Find best method for each metric
        if len(benchmark_results) > 1:
            # For model size (smaller is better)
            best_size_method = min(benchmark_results.items(), key=lambda x: x[1]["model_size"]["size_gb"])[0]
            f.write(f"- **Model Size**: {best_size_method} provides the most compact model\n")
            
            # For inference speed (higher is better)
            best_speed_method = max(benchmark_results.items(), key=lambda x: x[1]["inference_speed"]["avg_tokens_per_second"])[0]
            f.write(f"- **Inference Speed**: {best_speed_method} offers the fastest inference\n")
            
            # For memory usage (smaller is better)
            if all("gpu_memory_allocated_gb" in r["memory_usage"] for r in benchmark_results.values()):
                best_memory_method = min(benchmark_results.items(), key=lambda x: x[1]["memory_usage"]["gpu_memory_allocated_gb"])[0]
                f.write(f"- **Memory Efficiency**: {best_memory_method} uses the least GPU memory\n")
            
            # For HumanEval (higher is better)
            if humaneval_available:
                best_humaneval_method = max(benchmark_results.items(), key=lambda x: x[1]["humaneval_results"]["pass@1"])[0]
                f.write(f"- **Code Generation**: {best_humaneval_method} performs best on the HumanEval benchmark\n")
        
        f.write("\n### Trade-offs\n\n")
        f.write("- Lower precision quantization (e.g., int4) generally results in smaller model size and lower memory usage, but may impact performance on certain tasks.\n")
        f.write("- Higher precision formats maintain more of the original model's capabilities but require more resources.\n")
        f.write("- The optimal quantization method depends on the specific deployment constraints and performance requirements.\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("Based on the benchmarking results, we can make the following recommendations:\n\n")
        
        # Generate recommendations based on results
        if humaneval_available and len(benchmark_results) > 1:
            # Find method with best balance
            methods = list(benchmark_results.keys())
            
            # Normalize metrics (1 is best, 0 is worst)
            model_sizes = [r["model_size"]["size_gb"] for r in benchmark_results.values()]
            normalized_sizes = [1 - ((size - min(model_sizes)) / (max(model_sizes) - min(model_sizes))) for size in model_sizes]
            
            speeds = [r["inference_speed"]["avg_tokens_per_second"] for r in benchmark_results.values()]
            normalized_speeds = [(speed - min(speeds)) / (max(speeds) - min(speeds)) for speed in speeds]
            
            humaneval_scores = [r["humaneval_results"]["pass@1"] for r in benchmark_results.values()]
            normalized_humaneval = [(score - min(humaneval_scores)) / (max(humaneval_scores) - min(humaneval_scores)) for score in humaneval_scores]
            
            # Calculate combined score
            combined_scores = [(s + h + z) / 3 for s, h, z in zip(normalized_speeds, normalized_humaneval, normalized_sizes)]
            best_overall_idx = combined_scores.index(max(combined_scores))
            best_overall_method = methods[best_overall_idx]
            
            f.write(f"- **Best Overall Balance**: {best_overall_method} provides the best balance between model size, inference speed, and HumanEval performance.\n")
            
            # Find best for specific scenarios
            f.write(f"- **For Resource-Constrained Environments**: {min(benchmark_results.items(), key=lambda x: x[1]['model_size']['size_gb'])[0]} is recommended due to its minimal size and memory footprint.\n")
            f.write(f"- **For Performance-Critical Applications**: {max(benchmark_results.items(), key=lambda x: x[1]['humaneval_results']['pass@1'])[0]} maintains the highest code generation accuracy.\n")
            f.write(f"- **For Latency-Sensitive Applications**: {max(benchmark_results.items(), key=lambda x: x[1]['inference_speed']['avg_tokens_per_second'])[0]} offers the fastest response times.\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("Detailed visualizations are available in the results directory:\n\n")
        f.write("- `model_size_comparison.png`: Comparison of model sizes across quantization methods\n")
        f.write("- `inference_speed_comparison.png`: Comparison of inference speeds\n")
        f.write("- `memory_usage_comparison.png`: Comparison of GPU memory usage\n")
        if humaneval_available:
            f.write("- `normalized_metrics_heatmap.png`: Heatmap of normalized performance metrics\n")
            f.write("- `performance_tradeoffs.html`: Interactive Plotly visualization of performance trade-offs\n")
    
    logger.info(f"Report generated at {report_path}")

if __name__ == "__main__":
    main() 