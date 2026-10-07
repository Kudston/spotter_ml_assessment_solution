from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the assessment PDF using holdout metrics and the scorer chart."
    )
    parser.add_argument("--metrics", type=Path, default=Path("holdout_metrics.json"))
    parser.add_argument(
        "--chart",
        type=Path,
        default=Path("scorer_results/candidate_december.png"),
    )
    parser.add_argument("--output", type=Path, default=Path("assessment_report.pdf"))
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not args.chart.is_file():
        raise FileNotFoundError(
            f"Scorer chart not found: {args.chart}. Run score.py before building the report."
        )

    holdout = metrics["holdout"]
    improvement = metrics["mae_improvement_over_mileage_baseline_percent"]
    with PdfPages(args.output) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.text(0.08, 0.94, "Freight Rate Prediction Assessment", fontsize=20, weight="bold")
        figure.text(0.08, 0.90, "Validation approach and model performance", fontsize=13, color="#064A56")
        figure.text(
            0.08,
            0.84,
            (
                "Model: LightGBM absolute-error regression in a scikit-learn Pipeline. "
                "The pipeline imputes missing numeric values, one-hot encodes pickup, "
                "delivery, and equipment with unknown-category handling, and engineers "
                "calendar and coordinate-delta features. The load identifier is excluded."
            ),
            fontsize=10.5,
            wrap=True,
            va="top",
        )
        figure.text(
            0.08,
            0.72,
            (
                f"Split: chronological holdout. Train rows: {metrics['training_rows']:,} "
                f"(through {metrics['training_end']}); holdout rows: {metrics['holdout_rows']:,} "
                f"({metrics['holdout_start']} to {metrics['holdout_end']}). "
                "This holds out the last two complete months to approximate future-month "
                "prediction and prevent random temporal leakage."
            ),
            fontsize=10.5,
            wrap=True,
            va="top",
        )

        table_rows = [
            ["Holdout MAE", f"${holdout['mae']:,.2f}"],
            ["Holdout RMSE", f"${holdout['rmse']:,.2f}"],
            ["Holdout R²", f"{holdout['r2']:.4f}"],
            ["Holdout MAPE", f"{holdout['mape_percent']:.2f}%"],
            ["Mileage-only baseline MAE", f"${metrics['mileage_baseline_mae']:,.2f}"],
            ["MAE improvement vs baseline", f"{improvement:.1f}%"],
        ]
        table_axis = figure.add_axes([0.08, 0.44, 0.84, 0.20])
        table_axis.axis("off")
        table_axis.table(
            cellText=table_rows,
            colLabels=["Metric", "Result"],
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.65, 0.35],
        )
        figure.text(
            0.08,
            0.37,
            (
                f"Data checks: {metrics['training_missing_weight']:,} training weights and "
                f"{metrics['training_missing_market_index']:,} market-index values are missing "
                f"and imputed within the training pipeline. "
                f"{metrics['training_negative_weight']:,} weights are negative; they are retained "
                "as observed inputs rather than silently altered."
            ),
            fontsize=10.5,
            wrap=True,
            va="top",
        )
        figure.text(
            0.08,
            0.27,
            (
                f"Final outputs: {metrics['validation_rows']:,} validation predictions and "
                f"{metrics['december_rows']} December chart predictions. For December, market "
                "signals and coordinates are absent from the supplied inputs; the same pipeline "
                "handles them using training-derived median imputation."
            ),
            fontsize=10.5,
            wrap=True,
            va="top",
        )
        figure.text(
            0.08,
            0.10,
            "Validation metrics are from the held-out labeled training data; no final validation labels were used.",
            fontsize=9,
            color="#455A60",
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        chart = mpimg.imread(args.chart)
        figure, axis = plt.subplots(figsize=(11.69, 8.27))
        axis.imshow(chart)
        axis.axis("off")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    print(f"Saved assessment report: {args.output}")


if __name__ == "__main__":
    main()
