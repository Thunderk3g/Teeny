@echo off
echo Installing dependencies for LLM quantization benchmarking tools...
echo This may take a few minutes...

pip install -r benchmark_requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo Error installing dependencies from requirements file. 
    echo Attempting to install key packages individually...
    echo.
    
    pip install torch
    pip install transformers
    pip install numpy
    pip install matplotlib
    pip install pandas
    pip install psutil
    pip install huggingface-hub
    pip install accelerate
    pip install scikit-learn
    pip install scipy
    pip install tqdm
)

echo.
if %ERRORLEVEL% neq 0 (
    echo There were errors during installation. Please check the error messages above.
) else (
    echo Dependencies installed successfully!
    echo You can now run the benchmarking scripts.
)

pause 