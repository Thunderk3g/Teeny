#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparison script for multiple quantized LLMs
Runs benchmarks on multiple models and creates comparative visualizations
"""

import os
import argparse
import json
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple quantized models")
    parser.add_argument(
        "--model_paths",
        nargs="+",
        required=True,
        help="Paths to the quantized models or model IDs from Hugging Face Hub"
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        help="Human-readable names for the models (if not provided, will use last part of model path)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./comparison_results",
        help="Directory to save comparison results"
    )
    parser.add_argument(
        "--run_humaneval",
        action="store_true",
        help="Whether to run HumanEval benchmark for each model"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to generate for measuring inference speed"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use for computation"
    )
    
    return parser.parse_args()

def run_benchmark_for_model(model_path, output_dir, device="cuda", n_samples=5, run_humaneval=False):
    """Run benchmark script for a single model"""
    logger.info(f"Benchmarking model: {model_path}")
    
    cmd = [
        "python", "benchmark_quantized_models.py",
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--device", device,
        "--n_samples", str(n_samples),
        "--save_results"
    ]
    
    if run_humaneval:
        cmd.append("--run_humaneval")
    
    # Run the benchmark script as a subprocess
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        logger.error(f"Error benchmarking {model_path}:")
        logger.error(stderr)
        return None
    
    # Find the results file
    model_name = os.path.basename(model_path.rstrip("/"))
    result_file = os.path.join(output_dir, f"{model_name}_benchmark_results.json")
    
    if not os.path.exists(result_file):
        logger.error(f"Result file not found for {model_path}")
        return None
    
    # Load results
    with open(result_file, "r") as f:
        results = json.load(f)
    
    logger.info(f"Benchmark completed for {model_path}")
    return results

def load_results(output_dir, model_paths):
    """Load benchmark results for all models"""
    all_results = []
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path.rstrip("/"))
        result_file = os.path.join(output_dir, f"{model_name}_benchmark_results.json")
        
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                results = json.load(f)
                all_results.append(results)
        else:
            logger.warning(f"No results found for {model_path}")
    
    return all_results

def create_comparison_visualizations(all_results, model_names=None, output_dir="./comparison_results"):
    """Create comparison visualizations for all models"""
    if not all_results:
        logger.error("No results to visualize")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract model names if not provided
    if not model_names or len(model_names) != len(all_results):
        model_names = [r["model_name"] for r in all_results]
    
    # Create dataframe for comparison
    comparison_data = {
        "Model": model_names,
        "Size (GB)": [r["model_size"]["size_gb"] for r in all_results],
        "Parameters (billions)": [r["model_size"]["param_count_billions"] for r in all_results],
        "Tokens per second": [r["inference_speed"]["avg_tokens_per_second"] for r in all_results],
        "Average generation time (s)": [r["inference_speed"]["avg_generation_time"] for r in all_results],
        "CPU Memory (GB)": [r["memory_usage"]["cpu_memory_gb"] for r in all_results],
    }
    
    # Add GPU memory if available
    if "gpu_memory_allocated_gb" in all_results[0]["memory_usage"]:
        comparison_data["GPU Memory Allocated (GB)"] = [
            r["memory_usage"].get("gpu_memory_allocated_gb", 0) for r in all_results
        ]
    
    # Add HumanEval results if available
    humaneval_available = all(
        "humaneval_results" in r and "pass@1" in r.get("humaneval_results", {}) 
        for r in all_results
    )
    
    if humaneval_available:
        comparison_data["HumanEval Pass@1"] = [
            r["humaneval_results"]["pass@1"] for r in all_results
        ]
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save comparison data
    df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Create visualizations
    
    # Model size comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, df["Size (GB)"], color="skyblue")
    plt.title("Model Size Comparison")
    plt.ylabel("Size (GB)")
    plt.xticks(rotation=45, ha="right")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_size_comparison.png"))
    
    # Inference speed comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, df["Tokens per second"], color="lightgreen")
    plt.title("Inference Speed Comparison")
    plt.ylabel("Tokens per second")
    plt.xticks(rotation=45, ha="right")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_speed_comparison.png"))
    
    # Memory usage comparison (stacked bar for CPU/GPU)
    plt.figure(figsize=(12, 6))
    
    if "GPU Memory Allocated (GB)" in df.columns:
        cpu_memory = df["CPU Memory (GB)"]
        gpu_memory = df["GPU Memory Allocated (GB)"]
        
        p1 = plt.bar(model_names, cpu_memory, color="lightblue", label="CPU Memory")
        p2 = plt.bar(model_names, gpu_memory, bottom=cpu_memory, color="coral", label="GPU Memory")
        
        # Add value labels
        for i, (cpu, gpu) in enumerate(zip(cpu_memory, gpu_memory)):
            plt.text(i, cpu/2, f'{cpu:.1f}', ha='center', va='center', fontsize=9, color='black')
            plt.text(i, cpu + gpu/2, f'{gpu:.1f}', ha='center', va='center', fontsize=9, color='black')
        
        plt.legend()
    else:
        bars = plt.bar(model_names, df["CPU Memory (GB)"], color="lightblue")
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.title("Memory Usage Comparison")
    plt.ylabel("Memory (GB)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_usage_comparison.png"))
    
    # HumanEval results if available
    if humaneval_available:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, df["HumanEval Pass@1"], color="orange")
        plt.title("HumanEval Pass@1 Comparison")
        plt.ylabel("Pass@1 Score")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "humaneval_comparison.png"))
    
    # Create a "combined score" (normalized average of inference speed, model size inverse, and HumanEval if available)
    normalized_speed = (df["Tokens per second"] - df["Tokens per second"].min()) / (df["Tokens per second"].max() - df["Tokens per second"].min()) if len(df) > 1 else np.ones(len(df))
    
    # For size, smaller is better, so we invert the normalization
    normalized_size = 1 - (df["Size (GB)"] - df["Size (GB)"].min()) / (df["Size (GB)"].max() - df["Size (GB)"].min()) if len(df) > 1 else np.ones(len(df))
    
    if humaneval_available:
        normalized_humaneval = (df["HumanEval Pass@1"] - df["HumanEval Pass@1"].min()) / (df["HumanEval Pass@1"].max() - df["HumanEval Pass@1"].min()) if len(df) > 1 else np.ones(len(df))
        combined_score = (normalized_speed + normalized_size + normalized_humaneval) / 3
    else:
        combined_score = (normalized_speed + normalized_size) / 2
    
    # Plot combined score
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, combined_score, color="purple")
    plt.title("Combined Efficiency Score (higher is better)")
    plt.ylabel("Score (0-1)")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_score.png"))
    
    # Create a radar chart comparing all metrics
    metrics = ["Inference Speed", "Size Efficiency", "Memory Efficiency"]
    if humaneval_available:
        metrics.append("HumanEval Score")
    
    # Create data for radar chart
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, model in enumerate(model_names):
        values = [normalized_speed[i], normalized_size[i], 
                  1 - (df["CPU Memory (GB)"][i] - df["CPU Memory (GB)"].min()) / (df["CPU Memory (GB)"].max() - df["CPU Memory (GB)"].min()) if len(df) > 1 else 1]
        
        if humaneval_available:
            values.append(normalized_humaneval[i])
        
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels and title
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison", size=15, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_comparison.png"))
    
    logger.info(f"Comparison visualizations saved to {output_dir}")
    
    # Create a markdown summary
    with open(os.path.join(output_dir, "comparison_summary.md"), "w") as f:
        f.write("# Quantized Model Comparison Summary\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Models Compared\n\n")
        for i, model in enumerate(model_names):
            f.write(f"{i+1}. **{model}**\n")
            f.write(f"   - Size: {df['Size (GB)'][i]:.2f} GB\n")
            f.write(f"   - Parameters: {df['Parameters (billions)'][i]:.2f} billion\n")
            f.write(f"   - Inference Speed: {df['Tokens per second'][i]:.2f} tokens/sec\n")
            if humaneval_available:
                f.write(f"   - HumanEval Pass@1: {df['HumanEval Pass@1'][i]:.4f}\n")
            f.write("\n")
        
        f.write("## Performance Comparison\n\n")
        f.write("| Model | Size (GB) | Parameters (B) | Tokens/sec | ")
        if "GPU Memory Allocated (GB)" in df.columns:
            f.write("GPU Memory (GB) | ")
        f.write("CPU Memory (GB) | ")
        if humaneval_available:
            f.write("HumanEval | ")
        f.write("Combined Score |\n")
        
        f.write("|" + "---|" * (7 + (1 if humaneval_available else 0) + (1 if "GPU Memory Allocated (GB)" in df.columns else 0)) + "\n")
        
        for i, model in enumerate(model_names):
            f.write(f"| {model} | {df['Size (GB)'][i]:.2f} | {df['Parameters (billions)'][i]:.2f} | {df['Tokens per second'][i]:.2f} | ")
            if "GPU Memory Allocated (GB)" in df.columns:
                f.write(f"{df['GPU Memory Allocated (GB)'][i]:.2f} | ")
            f.write(f"{df['CPU Memory (GB)'][i]:.2f} | ")
            if humaneval_available:
                f.write(f"{df['HumanEval Pass@1'][i]:.4f} | ")
            f.write(f"{combined_score[i]:.3f} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Best for inference speed
        best_speed_idx = df["Tokens per second"].idxmax()
        f.write(f"- **Best for inference speed**: {model_names[best_speed_idx]} ({df['Tokens per second'][best_speed_idx]:.2f} tokens/sec)\n")
        
        # Most compact model
        best_size_idx = df["Size (GB)"].idxmin()
        f.write(f"- **Most compact model**: {model_names[best_size_idx]} ({df['Size (GB)'][best_size_idx]:.2f} GB)\n")
        
        # Lowest memory usage
        best_mem_idx = df["CPU Memory (GB)"].idxmin()
        f.write(f"- **Lowest CPU memory usage**: {model_names[best_mem_idx]} ({df['CPU Memory (GB)'][best_mem_idx]:.2f} GB)\n")
        
        if humaneval_available:
            # Best HumanEval performance
            best_humaneval_idx = df["HumanEval Pass@1"].idxmax()
            f.write(f"- **Best HumanEval performance**: {model_names[best_humaneval_idx]} (Pass@1: {df['HumanEval Pass@1'][best_humaneval_idx]:.4f})\n")
        
        # Best overall score
        best_score_idx = combined_score.argmax()
        f.write(f"- **Best overall score**: {model_names[best_score_idx]} (Score: {combined_score[best_score_idx]:.3f})\n")
    
    logger.info(f"Comparison summary saved to {output_dir}/comparison_summary.md")

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # If model names not provided, use base path
    if not args.model_names:
        args.model_names = [os.path.basename(path.rstrip("/")) for path in args.model_paths]
    
    # Ensure we have the same number of names as paths
    if len(args.model_names) != len(args.model_paths):
        logger.warning("Number of model names doesn't match number of model paths")
        args.model_names = [os.path.basename(path.rstrip("/")) for path in args.model_paths]
    
    # Run benchmarks for each model
    all_results = []
    for i, model_path in enumerate(args.model_paths):
        logger.info(f"Processing model {i+1}/{len(args.model_paths)}: {model_path}")
        
        # Check if we already have results for this model
        model_name = os.path.basename(model_path.rstrip("/"))
        result_file = os.path.join(args.output_dir, f"{model_name}_benchmark_results.json")
        
        if os.path.exists(result_file):
            logger.info(f"Loading existing results for {model_path}")
            with open(result_file, "r") as f:
                results = json.load(f)
                all_results.append(results)
        else:
            results = run_benchmark_for_model(
                model_path=model_path,
                output_dir=args.output_dir,
                device=args.device,
                n_samples=args.n_samples,
                run_humaneval=args.run_humaneval
            )
            
            if results:
                all_results.append(results)
    
    # Create comparison visualizations
    if all_results:
        create_comparison_visualizations(all_results, args.model_names, args.output_dir)
        logger.info("Model comparison completed successfully")
    else:
        logger.error("No benchmark results were collected")

if __name__ == "__main__":
    main() 