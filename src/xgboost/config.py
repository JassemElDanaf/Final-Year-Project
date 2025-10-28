from pathlib import Path
ROOT = Path(r"E:/FYP")
RAW  = ROOT / "data/raw"
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for p in (PROC, MODELS, REPORTS): p.mkdir(parents=True, exist_ok=True)
