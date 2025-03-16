#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for comparing quantization methods.
Creates charts displaying performance metrics across different quantization approaches.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize quantization benchmark results"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./quantized_models",
        help="Directory containing benchmark results"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualization_results",
        help="Directory to save visualization results"
    )
    
    parser.add_argument(
        "--comparison_type",
        type=str,
        choices=["methods", "models", "both"],
        default="both",
        help="Type of comparison to visualize"
    )
    
    return parser.parse_args()

def load_benchmark_results(results_dir: str) -> Dict:
    """Load benchmark results from the specified directory"""
    results = {}
    
    # Get all benchmark metric files
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file == "benchmark_metrics.json":
                model_path = Path(root)
                model_name = model_path.name
                
                # Extract model and quantization method
                parts = model_name.split('-')
                if len(parts) >= 2:
                    base_model = parts[0]
                    # Extract quantization method from the name
                    if 'fp16' in model_name:
                        quant_method = 'FP16'
                    elif 'int8' in model_name:
                        quant_method = 'Int8'
                    elif 'gptq' in model_name:
                        quant_method = 'GPTQ'
                    elif 'awq' in model_name:
                        quant_method = 'AWQ'
                    elif 'gguf' in model_name:
                        quant_method = 'GGUF'
                    else:
                        quant_method = 'Original'
                    
                    # Read benchmark results
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            data = json.load(f)
                            
                            # Store results
                            if base_model not in results:
                                results[base_model] = {}
                            
                            results[base_model][quant_method] = data
                    except Exception as e:
                        print(f"Error loading {os.path.join(root, file)}: {e}")
    
    return results

def visualize_model_size_comparison(results: Dict, output_dir: str):
    """Create model size comparison chart"""
    plt.figure(figsize=(12, 8))
    
    models = []
    methods = []
    sizes = []
    
    # Extract data
    for model, model_data in results.items():
        for method, data in model_data.items():
            if 'model_size' in data:
                models.append(model)
                methods.append(method)
                sizes.append(data['model_size']['size_gb'])
    
    # Create DataFrame-like structure
    unique_models = sorted(set(models))
    unique_methods = ['Original', 'FP16', 'Int8', 'GPTQ', 'AWQ', 'GGUF']
    unique_methods = [m for m in unique_methods if m in set(methods)]
    
    # Prepare data for grouped bar chart
    data = np.zeros((len(unique_models), len(unique_methods)))
    for i, model in enumerate(models):
        model_idx = unique_models.index(model)
        method_idx = unique_methods.index(methods[i])
        data[model_idx, method_idx] = sizes[i]
    
    # Plot grouped bar chart
    x = np.arange(len(unique_models))
    width = 0.8 / len(unique_methods)
    
    for i, method in enumerate(unique_methods):
        offset = (i - len(unique_methods) / 2 + 0.5) * width
        plt.bar(x + offset, data[:, i], width, label=method)
    
    # Add labels and title
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Model Size (GB)', fontsize=14)
    plt.title('Model Size Comparison Across Quantization Methods', fontsize=16)
    plt.xticks(x, unique_models, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, model in enumerate(unique_models):
        for j, method in enumerate(unique_methods):
            if data[i, j] > 0:
                offset = (j - len(unique_methods) / 2 + 0.5) * width
                plt.text(i + offset, data[i, j] + 0.1, f"{data[i, j]:.1f}", 
                         ha='center', va='bottom', fontsize=10)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_size_comparison.png'), dpi=300)
    plt.close()

def visualize_inference_speed_comparison(results: Dict, output_dir: str):
    """Create inference speed comparison chart"""
    plt.figure(figsize=(12, 8))
    
    models = []
    methods = []
    speeds = []
    
    # Extract data
    for model, model_data in results.items():
        for method, data in model_data.items():
            if 'inference_speed' in data:
                models.append(model)
                methods.append(method)
                speeds.append(data['inference_speed']['avg_tokens_per_second'])
    
    # Create DataFrame-like structure
    unique_models = sorted(set(models))
    unique_methods = ['Original', 'FP16', 'Int8', 'GPTQ', 'AWQ', 'GGUF']
    unique_methods = [m for m in unique_methods if m in set(methods)]
    
    # Prepare data for grouped bar chart
    data = np.zeros((len(unique_models), len(unique_methods)))
    for i, model in enumerate(models):
        model_idx = unique_models.index(model)
        method_idx = unique_methods.index(methods[i])
        data[model_idx, method_idx] = speeds[i]
    
    # Plot grouped bar chart
    x = np.arange(len(unique_models))
    width = 0.8 / len(unique_methods)
    
    for i, method in enumerate(unique_methods):
        offset = (i - len(unique_methods) / 2 + 0.5) * width
        plt.bar(x + offset, data[:, i], width, label=method)
    
    # Add labels and title
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Tokens per Second', fontsize=14)
    plt.title('Inference Speed Comparison Across Quantization Methods', fontsize=16)
    plt.xticks(x, unique_models, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, model in enumerate(unique_models):
        for j, method in enumerate(unique_methods):
            if data[i, j] > 0:
                offset = (j - len(unique_methods) / 2 + 0.5) * width
                plt.text(i + offset, data[i, j] + 0.1, f"{data[i, j]:.1f}", 
                         ha='center', va='bottom', fontsize=10)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_speed_comparison.png'), dpi=300)
    plt.close()

def visualize_memory_usage_comparison(results: Dict, output_dir: str):
    """Create memory usage comparison chart"""
    plt.figure(figsize=(12, 8))
    
    models = []
    methods = []
    memory = []
    
    # Extract data
    for model, model_data in results.items():
        for method, data in model_data.items():
            if 'memory_usage' in data:
                models.append(model)
                methods.append(method)
                memory.append(data['memory_usage']['memory_gb'])
    
    # Create DataFrame-like structure
    unique_models = sorted(set(models))
    unique_methods = ['Original', 'FP16', 'Int8', 'GPTQ', 'AWQ', 'GGUF']
    unique_methods = [m for m in unique_methods if m in set(methods)]
    
    # Prepare data for grouped bar chart
    data = np.zeros((len(unique_models), len(unique_methods)))
    for i, model in enumerate(models):
        model_idx = unique_models.index(model)
        method_idx = unique_methods.index(methods[i])
        data[model_idx, method_idx] = memory[i]
    
    # Plot grouped bar chart
    x = np.arange(len(unique_models))
    width = 0.8 / len(unique_methods)
    
    for i, method in enumerate(unique_methods):
        offset = (i - len(unique_methods) / 2 + 0.5) * width
        plt.bar(x + offset, data[:, i], width, label=method)
    
    # Add labels and title
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Memory Usage (GB)', fontsize=14)
    plt.title('Memory Usage Comparison Across Quantization Methods', fontsize=16)
    plt.xticks(x, unique_models, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, model in enumerate(unique_models):
        for j, method in enumerate(unique_methods):
            if data[i, j] > 0:
                offset = (j - len(unique_methods) / 2 + 0.5) * width
                plt.text(i + offset, data[i, j] + 0.1, f"{data[i, j]:.1f}", 
                         ha='center', va='bottom', fontsize=10)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage_comparison.png'), dpi=300)
    plt.close()

def visualize_method_comparison(results: Dict, output_dir: str):
    """Create comparison of each method across different metrics"""
    # Create a radar chart for each model to compare methods
    for model, model_data in results.items():
        # Skip if not enough data
        if len(model_data) < 2:
            continue
        
        # Set up metrics to compare
        metrics = [
            ('Model Size (GB)', 'model_size', 'size_gb', False),  # label, key1, key2, higher_is_better
            ('Memory Usage (GB)', 'memory_usage', 'memory_gb', False),
            ('Tokens per Second', 'inference_speed', 'avg_tokens_per_second', True),
            ('Generation Time (s)', 'inference_speed', 'avg_generation_time', False)
        ]
        
        # Get methods and normalize data
        methods = list(model_data.keys())
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Create a bar chart for each metric
        for i, (metric_label, key1, key2, higher_is_better) in enumerate(metrics):
            ax = axes[i]
            
            # Extract metric data
            metric_data = []
            for method in methods:
                if key1 in model_data[method] and key2 in model_data[method][key1]:
                    metric_data.append(model_data[method][key1][key2])
                else:
                    metric_data.append(0)
            
            # Plot bars
            bars = ax.bar(methods, metric_data)
            
            # Color bars based on performance (green for better, red for worse)
            if metric_data:
                normalized = np.array(metric_data) / max(metric_data) if max(metric_data) > 0 else np.zeros_like(metric_data)
                for j, bar in enumerate(bars):
                    if higher_is_better:
                        # Higher values are better (green)
                        bar.set_color(plt.cm.RdYlGn(normalized[j]))
                    else:
                        # Lower values are better (green)
                        bar.set_color(plt.cm.RdYlGn(1 - normalized[j]))
            
            # Add value labels
            for bar, value in zip(bars, metric_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}',
                        ha='center', va='bottom', fontsize=10)
            
            # Add labels
            ax.set_title(metric_label, fontsize=14)
            ax.set_ylabel(metric_label, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set the overall title
        fig.suptitle(f'Quantization Method Comparison for {model}', fontsize=18)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{model}_method_comparison.png'), dpi=300)
        plt.close()

def create_speedup_compression_chart(results: Dict, output_dir: str):
    """Create a scatter plot showing speedup vs compression for different methods"""
    plt.figure(figsize=(12, 10))
    
    # Prepare data
    models = []
    methods = []
    speedups = []
    compressions = []
    marker_sizes = []
    
    # Reference values for original models
    original_speed = {}
    original_size = {}
    
    # Find original values first
    for model, model_data in results.items():
        if 'Original' in model_data:
            if 'inference_speed' in model_data['Original']:
                original_speed[model] = model_data['Original']['inference_speed']['avg_tokens_per_second']
            if 'model_size' in model_data['Original']:
                original_size[model] = model_data['Original']['model_size']['size_gb']
    
    # Now calculate ratios
    for model, model_data in results.items():
        if model not in original_speed or model not in original_size:
            continue
            
        for method, data in model_data.items():
            if method == 'Original':
                continue
                
            if 'inference_speed' in data and 'model_size' in data:
                speed = data['inference_speed']['avg_tokens_per_second']
                size = data['model_size']['size_gb']
                
                # Calculate speedup and compression
                speedup = speed / original_speed[model]
                compression = original_size[model] / size
                
                models.append(model)
                methods.append(method)
                speedups.append(speedup)
                compressions.append(compression)
                # Use parameter count as marker size
                param_count = data['model_size']['param_count'] / 1e9  # in billions
                marker_sizes.append(param_count * 20)  # Scale appropriately
    
    # Create scatter plot
    unique_methods = list(set(methods))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_methods)))
    
    for i, method in enumerate(unique_methods):
        indices = [j for j, m in enumerate(methods) if m == method]
        plt.scatter(
            [compressions[j] for j in indices],
            [speedups[j] for j in indices],
            s=[marker_sizes[j] for j in indices],
            c=[colors[i] for _ in indices],
            alpha=0.7,
            label=method,
            edgecolors='black',
            linewidths=1
        )
    
    # Add model names as annotations
    for i, model in enumerate(models):
        plt.annotate(
            model,
            (compressions[i], speedups[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Add reference lines
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Compression Ratio (higher is better)', fontsize=14)
    plt.ylabel('Speedup Ratio (higher is better)', fontsize=14)
    plt.title('Quantization Methods: Speedup vs Compression', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    # Set log scale for better visualization
    plt.xscale('log')
    plt.yscale('log')
    
    # Add explanatory text
    plt.text(
        0.02, 0.02,
        "Marker size represents model parameter count (billions)",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='bottom'
    )
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_vs_compression.png'), dpi=300)
    plt.close()

def create_summary_dashboard(results: Dict, output_dir: str):
    """Create a summary dashboard of all quantization results"""
    plt.figure(figsize=(15, 12))
    
    # Set up GridSpec
    gs = plt.GridSpec(3, 3, figure=plt.gcf())
    
    # Collect all metrics
    all_models = []
    all_methods = []
    model_sizes = []
    mem_usages = []
    token_speeds = []
    gen_times = []
    
    # Extract data
    for model, model_data in results.items():
        for method, data in model_data.items():
            all_models.append(model)
            all_methods.append(method)
            
            # Model size
            if 'model_size' in data:
                model_sizes.append(data['model_size']['size_gb'])
            else:
                model_sizes.append(0)
            
            # Memory usage
            if 'memory_usage' in data:
                mem_usages.append(data['memory_usage']['memory_gb'])
            else:
                mem_usages.append(0)
            
            # Inference speed
            if 'inference_speed' in data:
                token_speeds.append(data['inference_speed']['avg_tokens_per_second'])
                gen_times.append(data['inference_speed']['avg_generation_time'])
            else:
                token_speeds.append(0)
                gen_times.append(0)
    
    # Model size comparison (top left)
    ax1 = plt.subplot(gs[0, 0])
    create_metric_miniplot(ax1, all_models, all_methods, model_sizes, 
                          'Model Size (GB)', cmap='viridis_r')  # reversed to make smaller better
    
    # Memory usage comparison (top center)
    ax2 = plt.subplot(gs[0, 1])
    create_metric_miniplot(ax2, all_models, all_methods, mem_usages, 
                          'Memory Usage (GB)', cmap='viridis_r')
    
    # Inference speed (top right)
    ax3 = plt.subplot(gs[0, 2])
    create_metric_miniplot(ax3, all_models, all_methods, token_speeds, 
                          'Tokens per Second')
    
    # Generation time (middle left)
    ax4 = plt.subplot(gs[1, 0])
    create_metric_miniplot(ax4, all_models, all_methods, gen_times, 
                          'Generation Time (s)', cmap='viridis_r')
    
    # Speedup vs. compression scatter (spans middle center and right)
    ax5 = plt.subplot(gs[1, 1:])
    create_embedded_scatter(ax5, results)
    
    # Method comparison heatmap (spans bottom row)
    ax6 = plt.subplot(gs[2, :])
    create_method_heatmap(ax6, results)
    
    # Set title
    plt.suptitle('Quantization Methods Comparison Dashboard', fontsize=20)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'quantization_dashboard.png'), dpi=300)
    plt.close()

def create_metric_miniplot(ax, models, methods, values, title, cmap='viridis'):
    """Create a small plot for a specific metric"""
    # Create DataFrame-like structure
    unique_models = sorted(set(models))
    unique_methods = ['Original', 'FP16', 'Int8', 'GPTQ', 'AWQ', 'GGUF']
    unique_methods = [m for m in unique_methods if m in set(methods)]
    
    # Prepare data for grouped bar chart
    data = np.zeros((len(unique_models), len(unique_methods)))
    for i, (model, method, value) in enumerate(zip(models, methods, values)):
        if model in unique_models and method in unique_methods:
            model_idx = unique_models.index(model)
            method_idx = unique_methods.index(method)
            data[model_idx, method_idx] = value
    
    # Plot heatmap instead of bars for compactness
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Add labels
    ax.set_title(title, fontsize=12)
    ax.set_yticks(np.arange(len(unique_models)))
    ax.set_yticklabels(unique_models, fontsize=10)
    ax.set_xticks(np.arange(len(unique_methods)))
    ax.set_xticklabels(unique_methods, fontsize=10, rotation=45, ha='right')
    
    # Add value text
    for i in range(len(unique_models)):
        for j in range(len(unique_methods)):
            if data[i, j] > 0:
                ax.text(j, i, f"{data[i, j]:.1f}", 
                       ha='center', va='center', 
                       color='white' if plt.cm.get_cmap(cmap)(data[i, j]/data.max()) < 0.5 else 'black',
                       fontsize=8)

def create_embedded_scatter(ax, results):
    """Create an embedded scatter plot for speedup vs compression"""
    # Prepare data
    models = []
    methods = []
    speedups = []
    compressions = []
    marker_sizes = []
    
    # Reference values for original models
    original_speed = {}
    original_size = {}
    
    # Find original values first
    for model, model_data in results.items():
        if 'Original' in model_data:
            if 'inference_speed' in model_data['Original']:
                original_speed[model] = model_data['Original']['inference_speed']['avg_tokens_per_second']
            if 'model_size' in model_data['Original']:
                original_size[model] = model_data['Original']['model_size']['size_gb']
    
    # Now calculate ratios
    for model, model_data in results.items():
        if model not in original_speed or model not in original_size:
            continue
            
        for method, data in model_data.items():
            if method == 'Original':
                continue
                
            if 'inference_speed' in data and 'model_size' in data:
                speed = data['inference_speed']['avg_tokens_per_second']
                size = data['model_size']['size_gb']
                
                # Calculate speedup and compression
                speedup = speed / original_speed[model]
                compression = original_size[model] / size
                
                models.append(model)
                methods.append(method)
                speedups.append(speedup)
                compressions.append(compression)
                
                # Use parameter count as marker size
                if 'param_count' in data['model_size']:
                    param_count = data['model_size']['param_count'] / 1e9  # in billions
                else:
                    param_count = 1  # default size
                marker_sizes.append(param_count * 20)  # Scale appropriately
    
    # Create scatter plot
    unique_methods = list(set(methods))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_methods)))
    
    for i, method in enumerate(unique_methods):
        indices = [j for j, m in enumerate(methods) if m == method]
        ax.scatter(
            [compressions[j] for j in indices],
            [speedups[j] for j in indices],
            s=[marker_sizes[j] for j in indices],
            c=[colors[i] for _ in indices],
            alpha=0.7,
            label=method,
            edgecolors='black',
            linewidths=1
        )
    
    # Add model names as annotations
    for i, model in enumerate(models):
        ax.annotate(
            model,
            (compressions[i], speedups[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Add reference lines
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    ax.set_xlabel('Compression Ratio (higher is better)', fontsize=12)
    ax.set_ylabel('Speedup Ratio (higher is better)', fontsize=12)
    ax.set_title('Speedup vs. Compression', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    
    # Set log scale for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')

def create_method_heatmap(ax, results):
    """Create a heatmap comparing methods across metrics"""
    # Prepare data structure
    methods = []
    metrics = ['Model Size (GB)', 'Memory Usage (GB)', 'Tokens/sec', 'Gen. Time (s)']
    data = []
    
    # Collect all unique methods
    for model_data in results.values():
        for method in model_data.keys():
            if method not in methods:
                methods.append(method)
    
    # Sort methods in a logical order
    ideal_order = ['Original', 'FP16', 'Int8', 'GPTQ', 'AWQ', 'GGUF']
    methods = [m for m in ideal_order if m in methods]
    
    # Initialize data matrix
    data = np.zeros((len(methods), len(metrics)))
    
    # Calculate average performance for each method across models
    counts = np.zeros((len(methods), len(metrics)))
    
    for model_data in results.values():
        for i, method in enumerate(methods):
            if method in model_data:
                # Model size
                if 'model_size' in model_data[method]:
                    data[i, 0] += model_data[method]['model_size']['size_gb']
                    counts[i, 0] += 1
                
                # Memory usage
                if 'memory_usage' in model_data[method]:
                    data[i, 1] += model_data[method]['memory_usage']['memory_gb']
                    counts[i, 1] += 1
                
                # Inference speed
                if 'inference_speed' in model_data[method]:
                    data[i, 2] += model_data[method]['inference_speed']['avg_tokens_per_second']
                    data[i, 3] += model_data[method]['inference_speed']['avg_generation_time']
                    counts[i, 2] += 1
                    counts[i, 3] += 1
    
    # Calculate averages
    for i in range(len(methods)):
        for j in range(len(metrics)):
            if counts[i, j] > 0:
                data[i, j] /= counts[i, j]
    
    # Normalize data (higher is always better)
    normalized = np.zeros_like(data)
    for j in range(len(metrics)):
        if np.max(data[:, j]) > 0:
            if j in [0, 1, 3]:  # For these metrics, lower is better
                # Invert for "lower is better" metrics
                normalized[:, j] = 1 - (data[:, j] / np.max(data[:, j]))
            else:
                normalized[:, j] = data[:, j] / np.max(data[:, j])
    
    # Create heatmap
    im = ax.imshow(normalized, cmap='RdYlGn', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.5, label='Normalized Performance (higher is better)')
    
    # Add labels
    ax.set_title('Quantization Methods Performance Comparison', fontsize=14)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=12)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=12)
    
    # Add value text
    for i in range(len(methods)):
        for j in range(len(metrics)):
            if counts[i, j] > 0:
                ax.text(j, i, f"{data[i, j]:.2f}", 
                       ha='center', va='center', 
                       color='white' if normalized[i, j] < 0.5 else 'black',
                       fontsize=10)
    
    # Add extra annotations
    ax.set_xlabel('Metrics (Model Size & Memory: lower is better, Speed: higher is better)', fontsize=12)
    ax.set_ylabel('Quantization Methods', fontsize=12)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load benchmark results
    results = load_benchmark_results(args.results_dir)
    
    if not results:
        print("No benchmark results found. Please run benchmarks first.")
        return 1
    
    print(f"Found benchmark results for {len(results)} models")
    
    # Create visualizations
    if args.comparison_type in ['models', 'both']:
        print("Creating model comparison visualizations...")
        visualize_model_size_comparison(results, args.output_dir)
        visualize_inference_speed_comparison(results, args.output_dir)
        visualize_memory_usage_comparison(results, args.output_dir)
    
    if args.comparison_type in ['methods', 'both']:
        print("Creating method comparison visualizations...")
        visualize_method_comparison(results, args.output_dir)
        create_speedup_compression_chart(results, args.output_dir)
    
    # Create summary dashboard
    print("Creating summary dashboard...")
    create_summary_dashboard(results, args.output_dir)
    
    print(f"Visualizations saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    exit(main()) 