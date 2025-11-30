#!/usr/bin/env python3
"""Aggregate OpenTuner CSV outputs and generate comparison figures."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


TECHNIQUES: Sequence[str] = (
    "PureRandom",
    "DifferentialEvolution",
    "RandomNelderMead",
    "PSO_GA_Bandit",
    "AUCBanditMetaTechniqueA",
)
EXPERIMENTS: Sequence[str] = ("rosenbrock", "mmm", "tsp", "gccflags")


def load_results(results_dir: Path) -> Dict[str, Dict[str, List[List[Tuple[int, float]]]]]:
    data: Dict[str, Dict[str, List[List[Tuple[int, float]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for csv_file in results_dir.glob("*.csv"):
        with csv_file.open() as handle:
            reader = csv.DictReader(handle)
            rows: List[Tuple[int, float]] = []
            experiment = None
            technique = None
            rep = None
            for row in reader:
                experiment = row["experiment"]
                technique = row["technique"]
                rep = row.get("rep")
                try:
                    generation = int(row["generation"])
                    best_time = float(row["best_time"])
                    rows.append((generation, best_time))
                except (ValueError, KeyError):
                    continue
            if experiment and technique and rows:
                rows.sort(key=lambda x: x[0])
                data[experiment][technique].append(rows)
                print(
                    f"Loaded {csv_file.name} ({experiment}, {technique}, rep={rep}) with {len(rows)} points"
                )
    return data


def aggregate_runs(runs: List[List[Tuple[int, float]]]):
    if not runs:
        return None
    all_generations = sorted({gen for run in runs for gen, _ in run})
    if not all_generations:
        return None

    padded = []
    for run in runs:
        idx = 0
        last = math.nan
        values: List[float] = []
        for gen in all_generations:
            while idx < len(run) and run[idx][0] <= gen:
                last = run[idx][1]
                idx += 1
            values.append(last)
        padded.append(values)

    arr = np.array(padded, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return all_generations, mean, std


def plot_convergence(ax, experiment: str, runs_by_tech: Dict[str, List[List[Tuple[int, float]]]], colors: Dict[str, tuple]):
    has_data = False
    for technique in TECHNIQUES:
        runs = runs_by_tech.get(technique, [])
        aggregated = aggregate_runs(runs)
        if aggregated is None:
            continue
        gens, mean, std = aggregated
        ax.plot(gens, mean, label=technique, color=colors[technique], linewidth=2)
        ax.fill_between(
            gens,
            mean - std,
            mean + std,
            color=colors[technique],
            alpha=0.2,
            linewidth=0,
        )
        has_data = True

    ax.set_title(f"{experiment.capitalize()} Convergence")
    ax.set_xlabel("Generation (new best discoveries)")
    ax.set_ylabel("Best Time")
    ax.grid(True, linestyle="--", alpha=0.5)
    if has_data:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")


def collect_final_scores(data: Dict[str, Dict[str, List[List[Tuple[int, float]]]]]):
    scores: Dict[str, List[float]] = {tech: [] for tech in TECHNIQUES}
    for experiment_runs in data.values():
        for technique, runs in experiment_runs.items():
            for run in runs:
                if run:
                    scores[technique].append(run[-1][1])
    return scores


def collect_final_scores_by_experiment(
    data: Dict[str, Dict[str, List[List[Tuple[int, float]]]]]
) -> Dict[str, Dict[str, List[float]]]:
    """Return final scores grouped per experiment and technique."""
    scores: Dict[str, Dict[str, List[float]]] = {}
    for experiment in data:
        scores[experiment] = {tech: [] for tech in TECHNIQUES}
        for technique, runs in data[experiment].items():
            for run in runs:
                if run:
                    scores[experiment][technique].append(run[-1][1])
    return scores


def ensure_dirs(figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "experiments" / "results"
    figures_dir = repo_root / "experiments" / "figures"

    ensure_dirs(figures_dir)

    data = load_results(results_dir)
    if not data:
        print("No CSV results found. Run experiments first.")
        return

    cmap = plt.get_cmap("tab10")
    colors = {tech: cmap(i % 10) for i, tech in enumerate(TECHNIQUES)}

    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: Rosenbrock convergence
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_convergence(ax1, "rosenbrock", data.get("rosenbrock", {}), colors)
    fig1.tight_layout()
    fig1.savefig(figures_dir / "fig1_convergence_rosenbrock.png", dpi=200)
    plt.close(fig1)

    # Figure 2: MMM convergence
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_convergence(ax2, "mmm", data.get("mmm", {}), colors)
    fig2.tight_layout()
    fig2.savefig(figures_dir / "fig2_convergence_mmm.png", dpi=200)
    plt.close(fig2)

    # Figure 3: TSP convergence
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    plot_convergence(ax3, "tsp", data.get("tsp", {}), colors)
    fig3.tight_layout()
    fig3.savefig(figures_dir / "fig3_convergence_tsp.png", dpi=200)
    plt.close(fig3)

    # Figure 4: GCCFlags convergence
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    plot_convergence(ax4, "gccflags", data.get("gccflags", {}), colors)
    fig4.tight_layout()
    fig4.savefig(figures_dir / "fig4_convergence_gccflags.png", dpi=200)
    plt.close(fig4)

    # Figure 5: Technique comparison per experiment (2x2 bar charts)
    scores_by_experiment = collect_final_scores_by_experiment(data)
    fig5, axes5 = plt.subplots(2, 2, figsize=(12, 10))
    for ax, experiment in zip(axes5.flat, EXPERIMENTS):
        experiment_scores = scores_by_experiment.get(
            experiment, {tech: [] for tech in TECHNIQUES}
        )
        labels = []
        means = []
        errors = []
        for tech in TECHNIQUES:
            scores = experiment_scores.get(tech, [])
            if not scores:
                continue
            labels.append(tech)
            means.append(float(np.mean(scores)))
            errors.append(float(np.std(scores)) / math.sqrt(len(scores)))
        if labels:
            positions = np.arange(len(labels))
            ax.bar(
                positions,
                means,
                yerr=errors,
                color=[colors[t] for t in labels],
                alpha=0.8,
                capsize=5,
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Final Best Time (mean ± SE)")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(experiment.capitalize())
    fig5.tight_layout()
    fig5.savefig(figures_dir / "fig5_technique_comparison.png", dpi=200)
    plt.close(fig5)

    # Figure 6: Box plots per experiment (2x2 grid)
    fig6, axes6 = plt.subplots(2, 2, figsize=(12, 10))
    for ax, experiment in zip(axes6.flat, EXPERIMENTS):
        experiment_scores = scores_by_experiment.get(
            experiment, {tech: [] for tech in TECHNIQUES}
        )
        box_data = [experiment_scores[tech] for tech in TECHNIQUES if experiment_scores.get(tech)]
        box_labels = [tech for tech in TECHNIQUES if experiment_scores.get(tech)]
        if box_data:
            box = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, tech in zip(box["boxes"], box_labels):
                patch.set_facecolor(colors[tech])
                patch.set_alpha(0.7)
            ax.set_xticklabels(box_labels, rotation=45, ha="right")
            ax.set_ylabel("Final Best Time")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(experiment.capitalize())
    fig6.tight_layout()
    fig6.savefig(figures_dir / "fig6_boxplot.png", dpi=200)
    plt.close(fig6)

    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
