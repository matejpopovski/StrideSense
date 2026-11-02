"""Draw the README figures from results/. Everything plotted is read from those files.

python -m ml.plots
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

RESULTS = Path("results")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SHORT = ("Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying")

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": SECONDARY,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def confusion(metrics: dict) -> None:
    matrix = np.array(metrics["litert"]["confusion"], dtype=float)
    share = matrix / matrix.sum(axis=1, keepdims=True)
    ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#2a78d6", "#0d366b"])
    figure, axes = plt.subplots(figsize=(6.4, 5.2), dpi=160)
    axes.imshow(share, cmap=ramp, vmin=0, vmax=1)
    for row, column in np.ndindex(*matrix.shape):
        if matrix[row, column]:
            color = "white" if share[row, column] > 0.45 else INK
            axes.text(
                column, row, f"{int(matrix[row, column])}", ha="center", va="center", color=color
            )
    axes.set_xticks(range(6), SHORT, rotation=35, ha="right")
    axes.set_yticks(range(6), SHORT)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True activity")
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_title(
        "Exported LiteRT model on the 9 held-out subjects, window counts",
        loc="left",
        fontsize=10.5,
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(RESULTS / "confusion_matrix.png")
    plt.close(figure)


def per_subject(metrics: dict) -> None:
    subjects = metrics["litert"]["per_subject"]
    ordered = sorted(subjects.items(), key=lambda item: item[1]["accuracy"])
    values = [entry["accuracy"] * 100 for _, entry in ordered]
    names = [f"Subject {subject}" for subject, _ in ordered]
    overall = metrics["litert"]["accuracy"] * 100
    figure, axes = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    axes.barh(names, values, color=BLUE, height=0.55)
    for index, value in enumerate(values):
        axes.text(value + 0.6, index, f"{value:.1f}%", va="center", fontsize=9, color=INK)
    axes.axvline(overall, color=SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    axes.text(
        overall + 1, 0, f"all windows {overall:.1f}%", fontsize=8.5, color=SECONDARY, va="center"
    )
    axes.set_xlim(0, 112)
    axes.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(length=0)
    axes.spines["left"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.set_title("Accuracy per held-out subject", loc="left", fontsize=10.5, color=INK)
    figure.tight_layout()
    figure.savefig(RESULTS / "per_subject.png")
    plt.close(figure)


def orientation(report: dict) -> None:
    figure, axes = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    styles = {"x": (BLUE, "X axis"), "y": (ORANGE, "Y axis"), "z": (AQUA, "Z axis")}
    shipped = report["models"]["shipped"]["per_axis"]
    for axis, (color, label) in styles.items():
        angles = [int(angle) for angle in shipped[axis]]
        values = [value * 100 for value in shipped[axis].values()]
        axes.plot(angles, values, color=color, linewidth=2, marker="o", markersize=4, label=label)
    chance = 100 / len(SHORT)
    axes.axhline(chance, color=SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    axes.text(2, chance - 2.5, "chance, 1 in 6", fontsize=8.5, color=SECONDARY, va="top")
    for label, x, y in (("X axis", 122, 47), ("Y axis", 63, 53), ("Z axis", 21, 54)):
        axes.text(x, y, label, fontsize=8.5, color=SECONDARY, va="center")
    axes.set_ylim(0, 100)
    axes.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    axes.set_xticks([0, 45, 90, 135, 180], ["0°", "45°", "90°", "135°", "180°"])
    axes.set_xlabel("Rotation of the sensor frame")
    axes.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes.tick_params(length=0)
    axes.spines["left"].set_visible(False)
    axes.legend(frameon=False, loc="upper right", labelcolor=SECONDARY)
    axes.set_title(
        "Shipped model on held-out windows, rotated (a proxy, not a pocket)",
        loc="left",
        fontsize=10.5,
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(RESULTS / "orientation.png")
    plt.close(figure)


def main() -> None:
    metrics = json.loads((RESULTS / "test_metrics.json").read_text())
    confusion(metrics)
    per_subject(metrics)
    orientation(json.loads((RESULTS / "orientation.json").read_text()))


if __name__ == "__main__":
    main()
