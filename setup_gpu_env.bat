@echo off
echo Setting up GPU environment for TFT model...

REM Activate virtual environment
call .venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other required packages
pip install pytorch-lightning
pip install pytorch-forecasting
pip install pandas numpy matplotlib scikit-learn

REM Install additional useful packages
pip install notebook
pip install tqdm
pip install optuna

echo Installation complete! Running GPU verification...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

pause