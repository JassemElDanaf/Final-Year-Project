# One-command launcher for TFT training on GPU machine
# Customize environment variables below as needed.

param(
    [int]$EncoderDays = 7,
    [int]$Hidden = 128,
    [int]$Heads = 4,
    [int]$Batch = 64,
    [int]$Epochs = 30,
    [string]$DryRun = "0"
)

. .\.venv\Scripts\Activate.ps1

$env:TFT_ENCODER_DAYS = "$EncoderDays"
$env:TFT_HIDDEN        = "$Hidden"
$env:TFT_HEADS         = "$Heads"
$env:TFT_BATCH         = "$Batch"
$env:TFT_EPOCHS        = "$Epochs"
$env:DRY_RUN           = "$DryRun"

Write-Host "Launching training with: DAYS=$EncoderDays HIDDEN=$Hidden HEADS=$Heads BATCH=$Batch EPOCHS=$Epochs DRY_RUN=$DryRun" -ForegroundColor Cyan
python src/tft/train_gpu.py
