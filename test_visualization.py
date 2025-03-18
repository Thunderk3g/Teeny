#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify visualization functions from benchmark_qwen_0.5b.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_benchmark_data():
    """Create mock benchmark data for visualization testing"""
    # Create sample benchmark results for three quantization methods
    benchmark_results = {
        "fp16": {
            "model_size": {
                "size_mb": 1024,  # 1 GB
                "size_gb": 1.0,
                "param_count": 500000000,
                "param_count_billions": 0.5
            },
            "memory_usage": {
                "cpu_memory_gb": 2.5,
                "gpu_memory_allocated_gb": 1.2,
                "gpu_memory_reserved_gb": 1.5
            },
            "inference_speed": {
                "avg_generation_time": 0.25,
                "avg_tokens_per_second": 40.0
            },
            "humaneval_results": {
                "pass@1": 0.35
            }
        },
        "int8": {
            "model_size": {
                "size_mb": 512,  # 0.5 GB
                "size_gb": 0.5,
                "param_count": 500000000,
                "param_count_billions": 0.5
            },
            "memory_usage": {
                "cpu_memory_gb": 2.2,
                "gpu_memory_allocated_gb": 0.8,
                "gpu_memory_reserved_gb": 1.0
            },
            "inference_speed": {
                "avg_generation_time": 0.30,
                "avg_tokens_per_second": 32.0
            },
            "humaneval_results": {
                "pass@1": 0.32
            }
        },
        "int4": {
            "model_size": {
                "size_mb": 256,  # 0.25 GB
                "size_gb": 0.25,
                "param_count": 500000000,
                "param_count_billions": 0.5
            },
            "memory_usage": {
                "cpu_memory_gb": 2.0,
                "gpu_memory_allocated_gb": 0.5,
                "gpu_memory_reserved_gb": 0.7
            },
            "inference_speed": {
                "avg_generation_time": 0.35,
                "avg_tokens_per_second": 28.0
            },
            "humaneval_results": {
                "pass@1": 0.28
            }
        }
    }
    
    return benchmark_results

def create_visualizations(benchmark_results, output_dir):
    """Create visualizations of benchmark results"""
    logger.info("Creating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center")
    
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
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center")
    
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

def generate_report(benchmark_results, output_dir):
    """Generate a comprehensive report"""
    report_path = os.path.join(output_dir, "benchmark_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Qwen 0.5B Quantization Benchmark Report\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report presents the benchmarking results for the Qwen 0.5B model under different quantization schemes.\n\n")
        
        f.write("## Model Information\n\n")
        f.write(f"- **Base Model**: Qwen/Qwen2.5-0.5B\n")
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

def main():
    """Main function"""
    output_dir = "./test_vis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mock benchmark data
    benchmark_results = create_mock_benchmark_data()
    
    # Create visualizations
    create_visualizations(benchmark_results, output_dir)
    
    # Generate report
    generate_report(benchmark_results, output_dir)
    
    # Save complete results as JSON
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        # Convert non-serializable objects to strings
        serializable_results = json.dumps(benchmark_results, default=lambda o: str(o), indent=4)
        f.write(serializable_results)
    
    logger.info(f"Test complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 