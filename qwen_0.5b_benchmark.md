# Qwen 0.5B Quantization Benchmark

This project benchmarks the quantized Qwen 0.5B model ("Qwen/Qwen2.5-0.5B") across various quantization methods, focusing on performance with the HumanEval benchmark.

## Features

- **Modular Architecture**: Follows the Teeny project structure for consistency
- **Multiple Quantization Methods**: Tests fp16, int8, and int4 quantization
- **Comprehensive Benchmarking**: Measures model size, inference speed, and memory usage
- **HumanEval Integration**: Tests functional correctness against the HumanEval benchmark
- **Rich Visualizations**: Uses Matplotlib, Seaborn, and Plotly for detailed performance charts

## Requirements

Install the requirements:

```bash
pip install torch transformers datasets matplotlib seaborn plotly psutil pandas
```

For HumanEval benchmarking, the script will automatically set up the bigcode-evaluation-harness when needed.

## Usage

Basic usage:

```bash
python benchmark_qwen_0.5b.py
```

With HumanEval benchmarking:

```bash
python benchmark_qwen_0.5b.py --run_humaneval
```

Custom configuration:

```bash
python benchmark_qwen_0.5b.py --methods fp16,int8,int4 --output_dir ./my_results --run_humaneval --humaneval_limit 10
```

## Parameters

- `--output_dir`: Directory to save benchmark results (default: "./qwen_0.5b_results")
- `--methods`: Comma-separated list of quantization methods to benchmark (default: "fp16,int8,int4")
- `--device`: Device to use for computation (default: "cuda" if available, otherwise "cpu")
- `--n_samples`: Number of samples to generate for measuring inference speed (default: 5)
- `--run_humaneval`: Whether to run HumanEval benchmark
- `--humaneval_limit`: Number of HumanEval problems to evaluate (default: 20)

## Output

The script generates multiple outputs:

1. **Performance metrics**: Model size, inference speed, memory usage, and HumanEval results
2. **Visualizations**: Bar charts, heatmaps, and interactive Plotly visualizations
3. **Detailed report**: A comprehensive markdown report with analysis and recommendations
4. **Raw data**: CSV and JSON files containing all benchmark data

## Analysis Workflow

1. The script loads and quantizes the model using different methods
2. It benchmarks each model variant using the same test prompts
3. If requested, it runs the HumanEval benchmark to test code generation capabilities
4. It generates comprehensive visualizations comparing the performance metrics
5. Finally, it creates a detailed report with findings and recommendations 