#!/usr/bin/env python3
"""Generate Lipophilicity subset thesis figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(
    "/Users/emirerdem/Library/Mobile Documents/com~apple~CloudDocs/"
    "Projects/M.Sc. Thesis"
)
SCAF = BASE / (
    "molnet_esol_project/benchmark_v5/results/lipophilicity_subset/"
    "lipophilicity_seed0_scaffold_subset.csv"
)
RAND = BASE / (
    "molnet_esol_project/benchmark_v5/results/lipophilicity_subset/"
    "lipophilicity_seed0_random_subset.csv"
)
OUT = BASE / "Thesis template/fig/results"
OUT.mkdir(parents=True, exist_ok=True)

# Display names used in thesis figures
DISPLAY = {
    "Combined + XGBoost": "Combined+XGB",
    "Descriptors + SVM": "Descriptors+SVM",
    "Descriptors + XGBoost": "Descriptors+XGB",
    "Morgan + XGBoost": "Morgan+XGB",
    "GIN (2D)": "GIN2D",
    "D-MPNN (2D)": "D-MPNN",
    "GIN (3D)": "GIN3D",
    "SchNet (3D coords)": "SchNet",
    "SMILES LSTM": "LSTM",
    "Morgan + RF": "Morgan+RF",
    "Morgan + SVM": "Morgan+SVM",
    "Descriptors + RF": "Descriptors+RF",
}

KEY_MODELS = [
    "Combined + XGBoost",
    "Descriptors + SVM",
    "Descriptors + XGBoost",
    "Morgan + XGBoost",
    "GIN (2D)",
    "D-MPNN (2D)",
    "GIN (3D)",
    "SchNet (3D coords)",
    "SMILES LSTM",
]

REL_MODELS = [
    "Combined + XGBoost",
    "Descriptors + SVM",
    "D-MPNN (2D)",
    "GIN (3D)",
    "SchNet (3D coords)",
]

ALL_MODELS = [
    "Combined + XGBoost",
    "Descriptors + SVM",
    "Descriptors + XGBoost",
    "Descriptors + RF",
    "Morgan + XGBoost",
    "Morgan + SVM",
    "Morgan + RF",
    "GIN (2D)",
    "D-MPNN (2D)",
    "GIN (3D)",
    "SchNet (3D coords)",
    "SMILES LSTM",
]

FRAC_PCT = {0.125: 12.5, 0.25: 25, 0.5: 50, 1.0: 100}
FRAC_ORDER = [0.125, 0.25, 0.5, 1.0]
X_PCT = [FRAC_PCT[f] for f in FRAC_ORDER]

# Colorblind-friendly qualitative palette
COLORS = {
    "Combined + XGBoost": "#0072B2",
    "Descriptors + SVM": "#E69F00",
    "Descriptors + XGBoost": "#56B4E9",
    "Morgan + XGBoost": "#009E73",
    "GIN (2D)": "#D55E00",
    "D-MPNN (2D)": "#CC79A7",
    "GIN (3D)": "#B8860B",  # dark goldenrod — readable on white
    "SchNet (3D coords)": "#666666",
    "SMILES LSTM": "#000000",
}
MARKERS = {
    "Combined + XGBoost": "o",
    "Descriptors + SVM": "s",
    "Descriptors + XGBoost": "^",
    "Morgan + XGBoost": "D",
    "GIN (2D)": "v",
    "D-MPNN (2D)": "P",
    "GIN (3D)": "X",
    "SchNet (3D coords)": "h",
    "SMILES LSTM": "*",
}
LS = {
    "Combined + XGBoost": "-",
    "Descriptors + SVM": "--",
    "Descriptors + XGBoost": "-.",
    "Morgan + XGBoost": ":",
    "GIN (2D)": "-",
    "D-MPNN (2D)": "--",
    "GIN (3D)": "-.",
    "SchNet (3D coords)": ":",
    "SMILES LSTM": "-",
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["note"] == "ok"].copy()
    return df


def series(df: pd.DataFrame, model: str) -> list[float]:
    sub = df[df["model"] == model].set_index("fraction")
    return [float(sub.loc[f, "RMSE"]) for f in FRAC_ORDER]


def apply_thesis_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def plot_learning_curves(scaf: pd.DataFrame, rand: pd.DataFrame, out: Path):
    apply_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)

    all_vals = []
    for ax, df, title in zip(
        axes,
        [scaf, rand],
        ["Scaffold split", "Random split"],
    ):
        for m in KEY_MODELS:
            y = series(df, m)
            all_vals.extend(y)
            ax.plot(
                X_PCT,
                y,
                color=COLORS[m],
                marker=MARKERS[m],
                linestyle=LS[m],
                linewidth=1.4,
                markersize=5.5,
                label=DISPLAY[m],
            )
        ax.set_xlabel("Training set size (%)")
        ax.set_xticks(X_PCT)
        ax.set_title(title)
        ax.set_xlim(8, 108)

    axes[0].set_ylabel("Test RMSE")
    ymin = min(all_vals) - 0.04
    ymax = max(all_vals) + 0.04
    axes[0].set_ylim(ymin, ymax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


def plot_relative(df: pd.DataFrame, out: Path, title: str):
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    for m in REL_MODELS:
        y_abs = series(df, m)
        y_rel = [v / y_abs[-1] for v in y_abs]
        ax.plot(
            X_PCT,
            y_rel,
            color=COLORS[m],
            marker=MARKERS[m],
            linestyle=LS[m],
            linewidth=1.5,
            markersize=6,
            label=DISPLAY[m],
        )

    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Training set size (%)")
    ax.set_ylabel(r"RMSE / RMSE$_{\mathrm{full}}$")
    ax.set_xticks(X_PCT)
    ax.set_xlim(8, 108)
    ax.set_title(title)
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


def plot_relative_scaffold(scaf: pd.DataFrame, out: Path):
    plot_relative(scaf, out, "Scaffold split — relative learning curves")


def plot_relative_random(rand: pd.DataFrame, out: Path):
    plot_relative(rand, out, "Random split — relative learning curves")


def latex_row(model: str, df: pd.DataFrame) -> str:
    vals = series(df, model)
    cells = " & ".join(f"{v:.3f}" for v in vals)
    return f"{DISPLAY.get(model, model)} & {cells} \\\\"


def best_at_each(df: pd.DataFrame, models=None):
    models = models or ALL_MODELS
    out = {}
    for f in FRAC_ORDER:
        sub = df[(df["fraction"] == f) & (df["model"].isin(models))]
        row = sub.loc[sub["RMSE"].idxmin()]
        out[FRAC_PCT[f]] = (row["model"], float(row["RMSE"]))
    return out


def main():
    scaf = load(SCAF)
    rand = load(RAND)

    p1 = OUT / "lipo_subset_learning_curves.png"
    p2 = OUT / "lipo_subset_relative_scaffold.png"
    p3 = OUT / "lipo_subset_relative_random.png"
    plot_learning_curves(scaf, rand, p1)
    plot_relative_scaffold(scaf, p2)
    plot_relative_random(rand, p3)

    print("\n=== LATEX: SCAFFOLD (all 12) ===")
    for m in ALL_MODELS:
        print(latex_row(m, scaf))

    print("\n=== LATEX: RANDOM (all 12) ===")
    for m in ALL_MODELS:
        print(latex_row(m, rand))

    print("\n=== LATEX: SCAFFOLD (9 key) ===")
    for m in KEY_MODELS:
        print(latex_row(m, scaf))

    print("\n=== LATEX: RANDOM (9 key) ===")
    for m in KEY_MODELS:
        print(latex_row(m, rand))

    print("\n=== BEST SCAFFOLD ===")
    for pct, (m, rmse) in best_at_each(scaf).items():
        print(f"  {pct}%: {DISPLAY.get(m, m)} ({rmse:.3f})")

    print("\n=== BEST RANDOM ===")
    for pct, (m, rmse) in best_at_each(rand).items():
        print(f"  {pct}%: {DISPLAY.get(m, m)} ({rmse:.3f})")

    print("\n=== Combined+XGB / D-MPNN wall time ===")
    for name, df in [("scaffold", scaf), ("random", rand)]:
        for m in ["Combined + XGBoost", "D-MPNN (2D)"]:
            for f in [0.125, 1.0]:
                row = df[(df["model"] == m) & (df["fraction"] == f)].iloc[0]
                print(
                    f"  {name} {DISPLAY[m]} @{FRAC_PCT[f]}%: "
                    f"RMSE={row['RMSE']:.4f}, seconds={row['seconds']:.3f}"
                )

    for p in [p1, p2, p3]:
        print(f"EXISTS={p.exists()} SIZE={p.stat().st_size} PATH={p}")


if __name__ == "__main__":
    main()
