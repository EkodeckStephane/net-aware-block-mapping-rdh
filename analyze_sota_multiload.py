"""Analyze multi-payload SOTA runs under matched-image protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import wilcoxon
from skimage.filters import threshold_otsu


METHODS = (
    "ABM-enumerative",
    "Dong-adaptive-conservative",
    "Huynh-Nguyen-T5",
    "PPOCP-conservative",
)


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def suitability(path: Path, threshold: int = 128) -> dict[str, float | bool]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    otsu = int(threshold_otsu(gray))
    fixed = gray >= threshold
    adaptive = gray >= otsu
    agreement = float(np.mean(fixed == adaptive))
    foreground = float(np.mean(fixed))
    return {
        "otsu_threshold": otsu,
        "fixed_otsu_agreement": agreement,
        "foreground_fraction": foreground,
        "suitable": agreement >= 0.95 and 0.05 <= foreground <= 0.95,
    }


def load_reports(paths: list[Path]) -> list[dict[str, object]]:
    reports = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            reports.append(json.load(handle))
    return sorted(reports, key=lambda report: int(report["payload_bits"]))


def analyze(
    reports: list[dict[str, object]],
    dataset: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_names = sorted(
        {
            str(row["image"])
            for report in reports
            for row in report["images"]
        }
    )
    suitability_by_image = {
        name: suitability(dataset / name) for name in image_names
    }
    curves = []
    paired_tests = []
    strata = []

    for report in reports:
        payload = int(report["payload_bits"])
        rows = list(report["images"])
        available = {
            method: {
                str(row["image"]): row
                for row in rows
                if row["method"] == method and row["available"]
            }
            for method in METHODS
        }
        matched = set.intersection(
            *(set(available[method]) for method in METHODS)
        )
        for method in METHODS:
            selected = [available[method][name] for name in sorted(matched)]
            if not selected:
                continue
            curves.append(
                {
                    "payload_bits": payload,
                    "method": method,
                    "matched_images": len(selected),
                    "mean_net_payload_bits": float(
                        np.mean([row["net_payload_bits"] for row in selected])
                    ),
                    "median_net_payload_bits": float(
                        np.median([row["net_payload_bits"] for row in selected])
                    ),
                    "median_drd": float(
                        np.median([row["drd"] for row in selected])
                    ),
                }
            )
        raw_p = []
        payload_tests = []
        for competitor in METHODS[1:]:
            common = sorted(
                set(available["ABM-enumerative"]) & set(available[competitor])
            )
            for metric in ("net_payload_bits", "drd"):
                abm = np.asarray(
                    [available["ABM-enumerative"][name][metric] for name in common],
                    dtype=float,
                )
                other = np.asarray(
                    [available[competitor][name][metric] for name in common],
                    dtype=float,
                )
                delta = abm - other
                try:
                    p_value = float(wilcoxon(delta).pvalue)
                except ValueError:
                    p_value = 1.0
                raw_p.append(p_value)
                payload_tests.append(
                    {
                        "payload_bits": payload,
                        "competitor": competitor,
                        "metric": metric,
                        "paired_images": len(common),
                        "median_abm_minus_competitor": float(np.median(delta)),
                        "wins": int(np.count_nonzero(delta > 0)),
                        "ties": int(np.count_nonzero(delta == 0)),
                        "wilcoxon_p": p_value,
                    }
                )
        for item, adjusted in zip(payload_tests, _holm(raw_p)):
            item["holm_adjusted_p"] = adjusted
        paired_tests.extend(payload_tests)

        for method in METHODS:
            method_rows = [
                row for row in rows if row["method"] == method and row["available"]
            ]
            for label, flag in (("suitable", True), ("unsuitable", False)):
                selected = [
                    row
                    for row in method_rows
                    if suitability_by_image[str(row["image"])]["suitable"] is flag
                ]
                if selected:
                    strata.append(
                        {
                            "payload_bits": payload,
                            "method": method,
                            "stratum": label,
                            "images": len(selected),
                            "availability_denominator": sum(
                                suitability_by_image[name]["suitable"] is flag
                                for name in image_names
                            ),
                            "mean_net_payload_bits": float(
                                np.mean(
                                    [row["net_payload_bits"] for row in selected]
                                )
                            ),
                            "median_drd": float(
                                np.median([row["drd"] for row in selected])
                            ),
                        }
                    )

    areas = []
    for method in METHODS:
        points = sorted(
            [row for row in curves if row["method"] == method],
            key=lambda row: row["median_drd"],
        )
        if len(points) >= 2:
            x = np.asarray([row["median_drd"] for row in points])
            y = np.asarray([row["mean_net_payload_bits"] for row in points])
            areas.append(
                {
                    "method": method,
                    "signed_trapezoidal_area": float(np.trapezoid(y, x)),
                    "note": "Descriptive only; each method spans a different DRD range.",
                }
            )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in METHODS:
        points = sorted(
            [row for row in curves if row["method"] == method],
            key=lambda row: row["payload_bits"],
        )
        if points:
            ax.plot(
                [row["median_drd"] for row in points],
                [row["mean_net_payload_bits"] for row in points],
                marker="o",
                label=method,
            )
            for row in points:
                ax.annotate(
                    str(row["payload_bits"]),
                    (row["median_drd"], row["mean_net_payload_bits"]),
                    fontsize=7,
                )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Median DRD on all-method matched images")
    ax.set_ylabel("Mean net payload (bits)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "net_bits_vs_drd.png", dpi=180)
    plt.close(fig)

    result = {
        "payloads": [int(report["payload_bits"]) for report in reports],
        "suitability_definition": (
            "Fixed threshold 128 agrees with Otsu on at least 95% of pixels "
            "and foreground fraction lies in [0.05, 0.95]."
        ),
        "suitable_images": int(
            sum(item["suitable"] for item in suitability_by_image.values())
        ),
        "image_suitability": suitability_by_image,
        "curves": curves,
        "areas": areas,
        "paired_wilcoxon_tests": paired_tests,
        "binarization_strata": strata,
    }
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("sota_multiload"))
    args = parser.parse_args()
    result = analyze(load_reports(args.reports), args.dataset, args.output_dir)
    print(
        json.dumps(
            {
                "payloads": result["payloads"],
                "suitable_images": result["suitable_images"],
                "areas": result["areas"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
