"""Evaluation charts (static PNGs). Colours: validated reference palette, light surface."""

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_confusion(metrics: dict, title: str, path) -> None:
    """2x2 confusion matrix with counts in the cells (sequential single-hue blue)."""
    grid = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max() * 1.15)
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            dark = grid[i, j] > grid.max() * 0.55
            ax.text(
                j,
                i,
                f"{grid[i, j]:,}\n{names[i][j]}",
                ha="center",
                va="center",
                color="white" if dark else INK,
                fontsize=10,
            )
    ax.set_xticks([0, 1], ["Predicted\nnot spam", "Predicted\nspam"], color=INK_SOFT)
    ax.set_yticks([0, 1], ["Actually\nnot spam", "Actually\nspam"], color=INK_SOFT)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(curves: dict, title: str, path) -> None:
    """Precision-recall curves for a few models; each line is labelled directly."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _style(ax)
    for color, (name, (y_true, scores)) in zip(SERIES, curves.items(), strict=False):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, color=color, linewidth=2, label=name)
    ax.set_xlabel("Recall (spam caught)", color=INK_SOFT)
    ax.set_ylabel("Precision (flagged that really are spam)", color=INK_SOFT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.01)
    ax.legend(frameon=False, labelcolor=INK_SOFT, loc="lower left")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(names, values, title, xlabel, path) -> None:
    """Horizontal bars (one series, so no legend); the value is printed at each bar end."""
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(names) + 1.6))
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.barh(names, values, color=BLUE, height=0.55)
    for y, v in enumerate(values):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(min(values) - 0.05 if min(values) > 0.5 else 0, 1.02)
    ax.set_xlabel(xlabel, color=INK_SOFT)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
