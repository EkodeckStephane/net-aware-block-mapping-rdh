"""Generate publication figures from the validated SOTA result files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "sota_multiload"
IMAGES = ROOT / "paper" / "images"
METHODS = (
    "ABM-enumerative",
    "Dong-adaptive-conservative",
    "Huynh-Nguyen-T5",
    "PPOCP-conservative",
)
LABELS = {
    "ABM-enumerative": "Proposed ABM",
    "Dong-adaptive-conservative": "Dong",
    "Huynh-Nguyen-T5": "Huynh-Nguyen",
    "PPOCP-conservative": "PPOCP",
}
COLORS = {
    "ABM-enumerative": "#006BA4",
    "Dong-adaptive-conservative": "#FF800E",
    "Huynh-Nguyen-T5": "#ABABAB",
    "PPOCP-conservative": "#595959",
}
LINESTYLES = {
    "ABM-enumerative": "-",
    "Dong-adaptive-conservative": "--",
    "Huynh-Nguyen-T5": "-.",
    "PPOCP-conservative": ":",
}
MARKERS = {
    "ABM-enumerative": "o",
    "Dong-adaptive-conservative": "s",
    "Huynh-Nguyen-T5": "^",
    "PPOCP-conservative": "D",
}


def load(name: str):
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def finish(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(IMAGES / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    analysis = load("analysis.json")
    curves = analysis["curves"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for method in METHODS:
        rows = sorted(
            [row for row in curves if row["method"] == method],
            key=lambda row: row["payload_bits"],
        )
        x = [row["payload_bits"] for row in rows]
        axes[0].plot(
            x,
            [row["mean_net_payload_bits"] for row in rows],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2,
            color=COLORS[method],
            label=LABELS[method],
        )
        axes[1].plot(
            x,
            [row["median_drd"] for row in rows],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2,
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Mean net payload (bits)")
    axes[1].set_ylabel("Median DRD")
    for axis in axes:
        axis.set_xlabel("Target payload (bits)")
        axis.set_xticks([64, 128, 256, 512])
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    finish(fig, "Figure_5")

    reports = [load(f"sota_{payload}.json") for payload in (64, 128, 256, 512)]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax = axes[0]
    x = np.arange(4)
    width = 0.19
    for index, method in enumerate(METHODS):
        availability = []
        for report in reports:
            summary = next(
                item for item in report["summaries"] if item["method"] == method
            )
            availability.append(100 * summary["availability_fraction"])
        ax.bar(
            x + (index - 1.5) * width,
            availability,
            width,
            label=LABELS[method],
            color=COLORS[method],
            edgecolor="black",
            hatch=("", "//", "\\\\", "xx")[index],
        )
    ax.set_xticks(x, ["64", "128", "256", "512"])
    ax.set_xlabel("Target payload (bits)")
    ax.set_ylabel("Availability (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    ax = axes[1]
    for method in METHODS:
        rows = sorted(
            [row for row in curves if row["method"] == method],
            key=lambda row: row["payload_bits"],
        )
        ax.plot(
            [row["median_drd"] for row in rows],
            [row["mean_net_payload_bits"] for row in rows],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2,
            color=COLORS[method],
            label=LABELS[method],
        )
        for row in rows:
            ax.annotate(
                str(row["payload_bits"]),
                (row["median_drd"], row["mean_net_payload_bits"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Median DRD on all-method matched images")
    ax.set_ylabel("Mean net payload (bits)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    finish(fig, "Figure_6")

    strata = [
        row
        for row in analysis["binarization_strata"]
        if row["method"] == "ABM-enumerative"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for label, color in (("suitable", "#006BA4"), ("unsuitable", "#ABABAB")):
        rows = sorted(
            [row for row in strata if row["stratum"] == label],
            key=lambda row: row["payload_bits"],
        )
        x_values = [row["payload_bits"] for row in rows]
        axes[0].plot(
            x_values,
            [
                100 * row["images"] / row["availability_denominator"]
                for row in rows
            ],
            marker="o" if label == "suitable" else "s",
            linestyle="-" if label == "suitable" else "--",
            linewidth=2,
            color=color,
            label=label.capitalize(),
        )
        axes[1].plot(
            x_values,
            [row["mean_net_payload_bits"] for row in rows],
            marker="o" if label == "suitable" else "s",
            linestyle="-" if label == "suitable" else "--",
            linewidth=2,
            color=color,
            label=label.capitalize(),
        )
    axes[0].set_ylabel("ABM availability (%)")
    axes[1].set_ylabel("Mean net payload (bits)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    for axis in axes:
        axis.set_xlabel("Target payload (bits)")
        axis.set_xticks([64, 128, 256, 512])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    finish(fig, "Figure_7")

    steganalysis = load("steganalysis.json")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for axis, classifier, title in (
        (axes[0], "logistic", "Logistic detector"),
        (axes[1], "random_forest", "Random Forest detector"),
    ):
        for method in METHODS:
            rows = sorted(
                [
                    row
                    for row in steganalysis
                    if row["method"] == method
                    and row["classifier"] == classifier
                ],
                key=lambda row: row["payload_bits"],
            )
            axis.plot(
                [row["payload_bits"] for row in rows],
                [row["roc_auc"] for row in rows],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                linewidth=2,
                color=COLORS[method],
                label=LABELS[method],
            )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Target payload (bits)")
        axis.set_ylabel("Grouped ROC AUC")
        axis.set_xticks([64, 128, 256, 512])
        axis.set_ylim(0.48, 1.0)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    finish(fig, "Figure_9")

    resources = load("resource_benchmark.json")["summaries"]
    resource_map = {row["method"]: row for row in resources}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    labels = [LABELS[method] for method in METHODS]
    colors = [COLORS[method] for method in METHODS]
    axes[0].bar(
        labels,
        [resource_map[method]["median_runtime_seconds"] for method in METHODS],
        color=colors,
        edgecolor="black",
        hatch=("", "//", "\\\\", "xx"),
    )
    axes[1].bar(
        labels,
        [
            resource_map[method]["median_peak_rss_increase_mib"]
            for method in METHODS
        ],
        color=colors,
        edgecolor="black",
        hatch=("", "//", "\\\\", "xx"),
    )
    axes[0].set_ylabel("Median runtime (s)")
    axes[1].set_ylabel("Median incremental peak RSS (MiB)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    finish(fig, "Figure_8")


if __name__ == "__main__":
    main()
