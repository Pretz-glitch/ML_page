"""Figures for the Phase 1 audit.

All functions write a PNG and close the figure; they never call ``plt.show``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import FEATURE_FAMILIES


def set_style() -> None:
    """Apply a small, consistent plotting style."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_correlation_heatmap(corr: pd.DataFrame, path: Path) -> None:
    """Save an annotated correlation heatmap."""
    set_style()
    labels = list(corr.columns)
    values = corr.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for row in range(len(labels)):
        for column in range(len(labels)):
            value = values[row, column]
            colour = "white" if abs(value) > 0.6 else "black"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=colour,
            )

    ax.set_title("Feature correlation across 26,085 station-days")
    fig.colorbar(image, ax=ax, shrink=0.82, label="Pearson r")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_variance_bars(variance: pd.DataFrame, path: Path) -> None:
    """Save a bar chart of the between-station variance fraction."""
    set_style()
    ordered = variance.sort_values("between_fraction", ascending=True)
    colours = [
        "#c0392b" if value >= 0.5 else "#2c7fb8"
        for value in ordered["between_fraction"]
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(ordered["feature"], ordered["between_fraction"], color=colours)
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fraction of variance that is between stations")
    ax.set_title("Between-station vs within-station variance")

    for position, value in enumerate(ordered["between_fraction"]):
        ax.text(
            min(value + 0.02, 0.98),
            position,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _tick_labels(dates: pd.Series, ticks: list[int]) -> list[str]:
    return [pd.Timestamp(dates.iloc[tick]).strftime("%Y-%m") for tick in ticks]


def save_coverage_figure(frame: pd.DataFrame, path: Path) -> None:
    """Save a two-panel figure: data availability and station x time structure."""
    set_style()
    availability = frame.pivot_table(
        index="station_id",
        columns="day_index",
        values="T_mean",
        aggfunc="count",
    )
    structure = frame.pivot(index="station_id", columns="day_index", values="T_mean")

    dates = (
        frame.drop_duplicates("day_index")
        .sort_values("day_index")
        .loc[:, "date"]
        .reset_index(drop=True)
    )
    ticks = [0, 120, 240, 360, 480, 600, frame["day_index"].max()]
    tick_labels = _tick_labels(dates, ticks)
    station_labels = [str(value) for value in availability.index]

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.6),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
        constrained_layout=True,
    )

    ax_left.imshow(
        availability.to_numpy(dtype=float),
        aspect="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax_left.set_title("Coverage: 37 stations x 705 days\n(all 26,085 station-days present)")
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Station")
    ax_left.set_xticks(ticks)
    ax_left.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax_left.set_yticks(range(0, len(station_labels), 4))
    ax_left.set_yticklabels(station_labels[::4], fontsize=7)

    image = ax_right.imshow(
        structure.to_numpy(dtype=float),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax_right.set_title("T_mean by station and day\n(shows seasonal and between-station structure)")
    ax_right.set_xlabel("Date")
    ax_right.set_ylabel("Station")
    ax_right.set_xticks(ticks)
    ax_right.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax_right.set_yticks(range(0, len(station_labels), 4))
    ax_right.set_yticklabels(station_labels[::4], fontsize=7)
    fig.colorbar(image, ax=ax_right, shrink=0.85, label="T_mean (normalised)")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_distributions(
    frame: pd.DataFrame, feature_names: tuple[str, ...], path: Path
) -> None:
    """Save a grid of feature histograms."""
    set_style()
    n_rows, n_columns = 3, 4
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(11.5, 7.2))
    axes_flat = np.asarray(axes).ravel()

    for axis, feature in zip(axes_flat, feature_names):
        values = frame[feature].to_numpy(dtype=float)
        family = FEATURE_FAMILIES.get(feature, "")
        axis.hist(values, bins=30, color="#2c7fb8", alpha=0.85, edgecolor="none")
        axis.axvline(float(np.median(values)), color="#c0392b", linewidth=1.2)
        axis.set_title(f"{feature}  ({family})", fontsize=9)
        axis.set_xlabel("normalised value")
        axis.set_ylabel("count")

    for axis in axes_flat[len(feature_names) :]:
        axis.axis("off")

    fig.suptitle("Feature distributions (raw normalised values, all station-days)", y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_detection_heatmap(
    auc: pd.DataFrame,
    recall: pd.DataFrame,
    path: Path,
    detector_names: dict[str, str] | None = None,
    fault_names: dict[str, str] | None = None,
) -> None:
    """Save a two-panel heatmap of mean ROC-AUC and recall@k."""
    set_style()
    detector_names = detector_names or {}
    fault_names = fault_names or {}

    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.4), constrained_layout=True
    )
    panels = [
        (auc, "ROC-AUC", 0.4, 1.0, axes[0]),
        (recall, "Recall at k", 0.0, 1.0, axes[1]),
    ]

    for frame, title, vmin, vmax, axis in panels:
        values = frame.to_numpy(dtype=float)
        image = axis.imshow(
            values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto"
        )

        axis.set_xticks(range(frame.shape[1]))
        axis.set_xticklabels(
            [fault_names.get(name, name) for name in frame.columns],
            rotation=30,
            ha="right",
        )
        axis.set_yticks(range(frame.shape[0]))
        axis.set_yticklabels(
            [detector_names.get(name, name) for name in frame.index]
        )
        axis.set_title(title)

        midpoint = 0.6 * (vmax - vmin) + vmin
        for row in range(frame.shape[0]):
            for column in range(frame.shape[1]):
                value = values[row, column]
                colour = "black" if value >= midpoint else "white"
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=colour,
                )

        fig.colorbar(image, ax=axis, shrink=0.82)

    fig.suptitle("Detection benchmark (mean over seeds and available magnitudes)")
    fig.text(0.5, -0.01, "* secondary detector", ha="center", va="top", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_faithfulness_heatmap(
    top1: pd.DataFrame,
    mrr: pd.DataFrame,
    path: Path,
    method_names: dict[str, str] | None = None,
    fault_names: dict[str, str] | None = None,
) -> None:
    """Save a two-panel heatmap of top-1 recovery and MRR."""
    set_style()
    method_names = method_names or {}
    fault_names = fault_names or {}

    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.4), constrained_layout=True
    )
    panels = [
        (top1, "Top-1 recovery", axes[0]),
        (mrr, "Mean reciprocal rank", axes[1]),
    ]

    for frame, title, axis in panels:
        values = frame.to_numpy(dtype=float)
        image = axis.imshow(
            values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto"
        )

        axis.set_xticks(range(frame.shape[1]))
        axis.set_xticklabels(
            [fault_names.get(name, name) for name in frame.columns],
            rotation=30,
            ha="right",
        )
        axis.set_yticks(range(frame.shape[0]))
        axis.set_yticklabels(
            [method_names.get(name, name) for name in frame.index]
        )
        axis.set_title(title)

        for row in range(frame.shape[0]):
            for column in range(frame.shape[1]):
                value = values[row, column]
                colour = "black" if value >= 0.6 else "white"
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=colour,
                )

        fig.colorbar(image, ax=axis, shrink=0.82)

    fig.suptitle("Attribution faithfulness (mean over seeds)")
    fig.text(
        0.5,
        -0.01,
        "chance top-1 ≈ 0.09 (1 of 11 features)",
        ha="center",
        va="top",
        fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

