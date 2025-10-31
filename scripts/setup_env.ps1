# Usage: Right-click -> Run with PowerShell (or run from an elevated PowerShell)
# This script prepares a Python venv, installs Git LFS files, and installs dependencies.
# It also installs PyTorch with a CUDA build you choose.

param(
    [string]$PythonVersion = "3.10",
    [string]$TorchCuda = "cu118" # options: 'cu118', 'cu121', or 'cpu'
)

Write-Host "[1/6] Ensuring Git LFS is available and pulling large files..." -ForegroundColor Cyan
try {
    git lfs version | Out-Null
} catch {
    Write-Error "Git LFS is not installed. Install from https://git-lfs.com/ and re-run."
    exit 1
}

git lfs install
# Pull LFS content (large CSV/models)
git lfs pull

Write-Host "[2/6] Creating Python virtual environment (.venv) ..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    py -$PythonVersion -m venv .venv
}

Write-Host "[3/6] Activating venv ..." -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "[4/6] Upgrading pip/setuptools/wheel ..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host "[5/6] Installing base requirements ..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "[6/6] Installing PyTorch ($TorchCuda) ..." -ForegroundColor Cyan
switch ($TorchCuda) {
    "cu118" { pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio }
    "cu121" { pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio }
    "cpu"   { pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu }
    default { pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio }
}

Write-Host "\nEnvironment ready. Next steps:" -ForegroundColor Green
Write-Host "- Activate: .\\.venv\\Scripts\\Activate.ps1"
Write-Host "- Optional: verify GPU: python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\""
Write-Host "- Train:   scripts\\train_tft.ps1"
