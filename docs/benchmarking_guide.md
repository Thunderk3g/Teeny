# LLM Quantization Benchmarking Tools

This directory contains scripts for benchmarking and comparing quantized large language models (LLMs).

## Scripts Overview

1. **benchmark_quantized_models.py** - Benchmark a single quantized model
2. **compare_quantized_models.py** - Compare multiple quantized models and generate visualizations

## Installation (IMPORTANT)

Before running any benchmark scripts, make sure you have all required dependencies installed:

```bash
# Install dependencies from the dedicated requirements file
pip install -r benchmark_requirements.txt
```

If you encounter module import errors, ensure you've installed all dependencies properly:

```bash
# Alternatively, install key packages individually
pip install torch transformers numpy matplotlib pandas psutil huggingface-hub accelerate
```

### Troubleshooting

- **ModuleNotFoundError: No module named 'torch'**: Run `pip install torch` to install PyTorch
- **Other missing modules**: Install the specific module using `pip install [module_name]`
- For GPU support, make sure you have installed PyTorch with CUDA support

## Single Model Benchmarking

The `benchmark_quantized_models.py` script measures:
- Model size and parameter count
- Memory usage (CPU and GPU)
- Inference speed (tokens per second)
- HumanEval benchmark performance (optional)

### Usage

```bash
python benchmark_quantized_models.py \
  --model_path "path/to/quantized/model" \
  --output_dir "./benchmark_results" \
  --device "cuda" \
  --n_samples 5 \
  --save_results \
  --visualize
```

To run with HumanEval benchmark (requires bigcode-evaluation-harness):

```bash
python benchmark_quantized_models.py \
  --model_path "path/to/quantized/model" \
  --output_dir "./benchmark_results" \
  --run_humaneval \
  --save_results \
  --visualize
```

### Parameters

- `--model_path`: Path to the quantized model or model ID from Hugging Face Hub (required)
- `--output_dir`: Directory to save benchmark results (default: "./benchmark_results")
- `--run_humaneval`: Whether to run HumanEval benchmark (requires bigcode-evaluation-harness)
- `--device`: Device to use for computation (default: "cuda" if available, otherwise "cpu")
- `--n_samples`: Number of samples to generate for measuring inference speed (default: 5)
- `--save_results`: Whether to save benchmark results to disk
- `--visualize`: Whether to create visualizations of benchmark results

## Multi-Model Comparison

The `compare_quantized_models.py` script benchmarks multiple models and creates comparative visualizations including:
- Bar charts for model size, memory usage, and inference speed
- Radar charts comparing all metrics
- Combined efficiency scores
- Detailed markdown summary report

### Usage

```bash
python compare_quantized_models.py \
  --model_paths "path/to/model1" "path/to/model2" "path/to/model3" \
  --model_names "GPTQ-4bit" "AWQ-4bit" "INT8" \
  --output_dir "./comparison_results" \
  --device "cuda" \
  --n_samples 5
```

With HumanEval benchmark:

```bash
python compare_quantized_models.py \
  --model_paths "path/to/model1" "path/to/model2" "path/to/model3" \
  --model_names "GPTQ-4bit" "AWQ-4bit" "INT8" \
  --output_dir "./comparison_results" \
  --run_humaneval
```

### Parameters

- `--model_paths`: Paths to the quantized models (required, can specify multiple)
- `--model_names`: Human-readable names for the models (optional, must match number of model_paths)
- `--output_dir`: Directory to save comparison results (default: "./comparison_results")
- `--run_humaneval`: Whether to run HumanEval benchmark for each model
- `--device`: Device to use for computation (default: "cuda")
- `--n_samples`: Number of samples to generate for measuring inference speed (default: 5)

## Output Files

### Single Model Benchmark

- JSON file with all benchmark results
- Visualizations for model size, memory usage, and inference speed

### Multi-Model Comparison

- CSV file with all comparison metrics
- Visualizations including:
  - Model size comparison (bar chart)
  - Inference speed comparison (bar chart) 
  - Memory usage comparison (stacked bar chart)
  - HumanEval Pass@1 comparison (if available)
  - Combined efficiency score (bar chart)
  - Radar chart for all metrics
- Markdown summary report with key findings

## Example Workflow

1. **First, install all dependencies**:
   ```bash
   pip install -r benchmark_requirements.txt
   ```

2. Quantize models using different methods (GPTQ, AWQ, INT8, etc.)

3. Benchmark each model individually:
   ```bash
   python benchmark_quantized_models.py --model_path "./quantized_models/qwen_gptq_4bit" --save_results
   ```

4. Compare all models:
   ```bash
   python compare_quantized_models.py --model_paths "./quantized_models/qwen_gptq_4bit" "./quantized_models/qwen_int8" "./quantized_models/qwen_awq_4bit"
   ```

5. View the comparison summary in the output directory:
   - `comparison_summary.md` - Text report with key findings
   - Various visualization PNG files
   - `model_comparison.csv` - Raw data for all metrics

## HumanEval Benchmark

The HumanEval benchmark evaluates a model's ability to generate functionally correct code for programming problems. When using the `--run_humaneval` option, you need to have the bigcode-evaluation-harness repository:

```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
pip install -e .
```

This benchmark measures Pass@1, which is the percentage of problems the model can solve correctly on the first attempt. 