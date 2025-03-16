# Project Structure Documentation

This document provides an overview of the project structure and explains the purpose of each component.

## Main Directory Structure

```
quantization-research/
├── quantize_llm.py              # Main script for quantizing models
├── visualize_quantization.py    # Script for creating visualizations of results
├── quantization_research.md     # Comprehensive overview of quantization techniques
├── README.md                    # Project overview and usage instructions
├── requirements.txt             # Core dependencies
├── benchmark_requirements.txt   # Additional dependencies for benchmarking
├── src/                         # Modular components and utilities
├── notebooks/                   # Jupyter notebooks for experimentation
├── docs/                        # Documentation files
├── quantized_models/            # Output directory for quantized models
├── visualization_results/       # Output directory for visualizations
└── Teeny/                       # Website files for project documentation
```

## Core Scripts

### `quantize_llm.py`

The main script for quantizing Large Language Models using various techniques. Supports multiple quantization methods including:

- GPTQ (2-4 bit precision)
- AWQ (4-bit precision)
- Int8 quantization (via BitsAndBytes)
- FP16 quantization

Features include:
- Model loading and preparation
- Calibration dataset handling
- Benchmarking of quantized models
- Optional HumanEval testing

### `visualize_quantization.py`

Script for creating visualizations of quantization results, including:

- Comparative charts for different quantization methods
- Model size, memory usage, and inference speed visualizations
- Performance retention metrics
- Summary dashboards with multiple metrics

## Source Code Directory (`src/`)

### `quantize_qwen.py`

Specialized script for quantizing Qwen2.5 models, with optimizations specific to this model architecture:

- Handles Qwen's transformer architecture with attention mechanism optimizations
- Supports GPTQ, AWQ, and BitsAndBytes quantization methods
- Includes calibration dataset generation
- Measures model size and inference speed

### `benchmark_quantized_models.py`

Standalone benchmarking tool for quantized models that evaluates:

- Inference speed (tokens/second)
- Memory usage during inference
- Model size on disk
- HumanEval benchmark performance (code generation capabilities)

### `compare_quantized_models.py`

Comparative analysis tool for multiple quantized models:

- Runs benchmarks on multiple models and creates comparative visualizations
- Generates side-by-side performance comparisons
- Creates visualizations showing tradeoffs between size, speed, and accuracy

## Notebooks

### `HumanEval_benchmark.ipynb`

Notebook for evaluating code generation capabilities of quantized models using the HumanEval benchmark:

- Runs the BigCode evaluation harness
- Tests model performance on coding tasks
- Analyzes and visualizes results
- Compares different quantization methods

## Documentation

### `quantization_research.md`

Comprehensive overview of quantization techniques and research findings, including:

- Detailed descriptions of quantization methods
- Precision formats and their characteristics
- Comparative metrics across different models and methods
- Model-specific considerations (particularly for Qwen models)

### `README.md`

Project overview and usage instructions, including:

- Installation steps
- Command-line examples for quantization and visualization
- Summary of research insights
- System requirements

## Dependencies

### `requirements.txt`

Core dependencies for the project, including:

- PyTorch for deep learning
- Transformers for model handling
- Visualization libraries (matplotlib, seaborn)
- Utility libraries (tqdm, psutil)

### `benchmark_requirements.txt`

Additional dependencies specifically for benchmarking, including:

- Data analysis tools (pandas, scikit-learn)
- Integration with Hugging Face Hub
- Additional visualization libraries 