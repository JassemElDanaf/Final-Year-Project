from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"E:/FYP")
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------- load summary ----------
summary = json.loads((ROOT / "models/agent1_accuracy_summary.json").read_text())

# case/format–insensitive key getter
def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

_normmap = { _norm(k): k for k in summary.keys() }

def getv(*cands, default=np.nan):
    for cand in cands:
        k = _normmap.get(_norm(cand))
        if k is not None:
            return summary[k]
    return default

# ---------- load per-house (accuracy preferred, else validation) ----------
ph_paths = [
    ROOT / "data/processed/agent1_accuracy_per_house.csv",
    ROOT / "data/processed/agent1_validation_per_house.csv",
]
for p in ph_paths:
    if p.exists():
        ph = pd.read_csv(p)
        break
else:
    ph = pd.DataFrame({"house_id": []})

# normalize per-house column names
ph.columns = [c.strip().lower() for c in ph.columns]

def find_col(cols, *must):
    must = [m.lower() for m in must]
    for c in cols:
        if all(m in c for m in must):
            return c
    return None

pv_mae_col        = find_col(ph.columns, "pv", "model", "mae") or find_col(ph.columns, "pv", "mae")
pv_base_mae_col   = find_col(ph.columns, "pv", "base", "mae")
load_mae_col      = find_col(ph.columns, "load", "model", "mae") or find_col(ph.columns, "load", "mae")
load_base_mae_col = find_col(ph.columns, "load", "base", "mae")

# ---------- Figure 1: headline KPIs ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

def kpi_panel(ax, title, mae_model, mae_base, smape_model, smape_base, r2_model, r2_base):
    labels = ["MAE (kW)", "sMAPE (%)"]
    model_vals = [mae_model, smape_model]
    base_vals  = [mae_base, smape_base]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, base_vals,  width=w, label="Baseline")
    ax.bar(x + w/2, model_vals, width=w, label="Model")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.96, f"R²: model {r2_model:.2f} | base {r2_base:.2f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    for i, v in enumerate(base_vals):
        ax.text(i - w/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(model_vals):
        ax.text(i + w/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

kpi_panel(
    axes[0], "PV (Day-Ahead)",
    getv("PV_MAE_kW"),
    getv("PV_base_MAE_kW"),
    getv("PV_sMAPE_%"),
    getv("PV_base_sMAPE_%"),
    getv("PV_R2"),
    getv("PV_base_R2")
)
kpi_panel(
    axes[1], "Load (Week-Ahead, Same Weekday)",
    getv("Load_MAE_kW"),
    getv("Load_base_MAE_kW"),
    getv("Load_sMAPE_%"),
    getv("Load_base_sMAPE_%"),
    getv("Load_R2"),
    getv("Load_base_R2")
)

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Agent-1 Forecast Quality (Model vs Baseline)", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0,0.07,1,0.95])
out1 = REPORTS / "agent1_kpis.png"
plt.savefig(out1, dpi=150); plt.close()

# ---------- Figure 2: per-house MAE bars ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

if "house_id" in ph.columns and not ph.empty:
    ph = ph.sort_values("house_id")

# PV panel
if {"house_id", pv_base_mae_col, pv_mae_col} <= set(ph.columns):
    axes[0].bar(ph["house_id"] - 0.15, ph[pv_base_mae_col], width=0.3, label="Baseline")
    axes[0].bar(ph["house_id"] + 0.15, ph[pv_mae_col],      width=0.3, label="Model")
    axes[0].set_ylabel("MAE (kW)")
    axes[0].set_title("PV — Per-House MAE (lower is better)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
else:
    axes[0].axis("off")
    axes[0].text(0.5, 0.5, "PV per-house MAE not available", ha="center", va="center")

# Load panel
if {"house_id", load_base_mae_col, load_mae_col} <= set(ph.columns):
    axes[1].bar(ph["house_id"] - 0.15, ph[load_base_mae_col], width=0.3, label="Baseline")
    axes[1].bar(ph["house_id"] + 0.15, ph[load_mae_col],      width=0.3, label="Model")
    axes[1].set_ylabel("MAE (kW)")
    axes[1].set_title("Load — Per-House MAE (lower is better)")
    axes[1].set_xlabel("House ID")
    axes[1].set_xticks(ph["house_id"])
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(frameon=False)
else:
    axes[1].axis("off")
    axes[1].text(0.5, 0.5, "Load per-house MAE not available", ha="center", va="center")

plt.tight_layout()
out2 = REPORTS / "agent1_per_house_mae.png"
plt.savefig(out2, dpi=150); plt.close()

print("Saved:")
print(" -", out1)
print(" -", out2)
