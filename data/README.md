# Data Folder

This repository tracks selected datasets via Git LFS to enable training on other machines.

- `processed/` contains the prepared hourly/15-min CSVs used by training.
- `raw/` is not included; place your raw house CSVs here if you want to regenerate.

If you performed a fresh clone:
```powershell
git lfs pull
```
This will fetch the large CSV files referenced by LFS.
