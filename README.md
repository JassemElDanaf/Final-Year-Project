# Final-Year-Project

Fast start guide to run training on a new (stronger) PC with GPU.

## 1) Clone with Git LFS
You must have Git and Git LFS installed.

```powershell
# install git lfs once on the machine (if not installed)
winget install Git.Git -e
winget install GitHub.GitLFS -e

# clone your repo
git clone https://github.com/JassemElDanaf/Final-Year-Project.git
cd Final-Year-Project

# pull LFS files (large CSVs / checkpoints)
git lfs install
git lfs pull
```

## 2) Create environment and install deps (Windows / PowerShell)
Use the helper script to set up a `.venv` and install PyTorch + packages.

```powershell
# Choose CUDA build that matches the GPU drivers on the machine
# Options: cu118, cu121, or cpu
scripts/setup_env.ps1 -PythonVersion 3.10 -TorchCuda cu118

# Verify
. .\.venv\Scripts\Activate.ps1
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3) Train TFT on GPU
Start with a faster config (finishes overnight). Adjust parameters as needed.

```powershell
# Example: 7-day encoder, smaller hidden size
scripts/train_tft.ps1 -EncoderDays 7 -Hidden 128 -Batch 64 -Epochs 30 -DryRun "0"
```

For maximum accuracy (slower):
```powershell
scripts/train_tft.ps1 -EncoderDays 14 -Hidden 160 -Batch 64 -Epochs 100 -DryRun "0"
```

## Project Layout
- `src/`           — Training code (TFT, baselines)
- `notebooks/`     — Analysis and reports
- `data/`          — Processed datasets (via Git LFS)
- `models/`        — Checkpoints and logs (via Git LFS)
- `scripts/`       — Setup and training launchers
- `requirements.txt` — Base Python dependencies

## Notes
- Large files are tracked with Git LFS. Ensure `git lfs pull` is run after clone.
- If training is slow, lower `-EncoderDays` to 7 and/or `-Hidden` to 96–128.
- If you hit Out-Of-Memory, try `-Batch 32`.

## Troubleshooting
- Update GPU drivers and CUDA runtime as needed.
- If `torch.cuda.is_available()` returns `False`, reinstall PyTorch with the correct CUDA build or pick `-TorchCuda cpu` for CPU-only.
