#!/usr/bin/env python3
"""Run OpenTuner experiments and export convergence traces to CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from opentuner.resultsdb import models


TECHNIQUES: Sequence[str] = (
    "PureRandom",
    "DifferentialEvolution",
    "RandomNelderMead",
    "PSO_GA_Bandit",  # PatternSearch has a bug, use PSO+GA bandit instead
    "AUCBanditMetaTechniqueA",
)


def _build_experiments(repo_root: Path) -> Dict[str, Dict[str, object]]:
    """Construct experiment metadata keyed by experiment name."""
    return {
        "rosenbrock": {
            "test_limit": 100,
            "working_dir": repo_root / "opentuner" / "examples" / "rosenbrock",
            "script_path": repo_root
            / "opentuner"
            / "examples"
            / "rosenbrock"
            / "rosenbrock.py",
        },
        "mmm": {
            "test_limit": 100,
            "working_dir": repo_root / "opentuner" / "examples" / "tutorials",
            "script_path": repo_root
            / "opentuner"
            / "examples"
            / "tutorials"
            / "mmm_tuner.py",
        },
        "tsp": {
            "test_limit": 100,
            "working_dir": repo_root / "opentuner" / "examples" / "tsp",
            "script_path": repo_root
            / "opentuner"
            / "examples"
            / "tsp"
            / "tsp.py",
            "extra_args": ["p01_d.txt"],  # Small problem (15 cities) - passed as positional arg
        },
        "gccflags": {
            "test_limit": 30,  # Reduced - each eval is slow (compile + run)
            "working_dir": repo_root / "opentuner" / "examples" / "gccflags",
            "script_path": repo_root
            / "opentuner"
            / "examples"
            / "gccflags"
            / "gccflags_minimal.py",
            "extra_args": [],
        },
    }


def run_command(cmd: List[str], cwd: Path) -> int:
    """Run a command, streaming output, and return the exit code."""
    print(f"Running: {' '.join(cmd)} (cwd={cwd})")
    completed = subprocess.run(cmd, cwd=str(cwd))
    return completed.returncode


def extract_convergence(db_path: Path) -> List[Tuple[int, float]]:
    """Pull (generation, best_time) pairs from an OpenTuner SQLite DB."""
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        tuning_run = (
            session.query(models.TuningRun)
            .order_by(models.TuningRun.id.desc())
            .first()
        )
        if tuning_run is None:
            print(f"No tuning runs found in {db_path}")
            return []

        query = (
            session.query(
                models.DesiredResult.generation,
                models.Result.time,
                models.Result.id,
            )
            .join(models.Result, models.DesiredResult.result_id == models.Result.id)
            .filter(models.DesiredResult.tuning_run_id == tuning_run.id)
            .filter(models.Result.was_new_best.is_(True))
            .filter(models.Result.state == "OK")
            .order_by(models.Result.id)
        )

        data: List[Tuple[int, float]] = []
        best = float("inf")
        for idx, (generation, time_value, _res_id) in enumerate(query, start=1):
            if time_value is None:
                continue
            best = min(best, time_value)
            generation_value = generation if generation is not None else idx
            data.append((int(generation_value), float(best)))

        # Fallback: if no "new best" records were found, use all results in order.
        if not data:
            fallback = (
                session.query(models.Result.time, models.Result.id)
                .filter(models.Result.tuning_run_id == tuning_run.id)
                .filter(models.Result.state == "OK")
                .order_by(models.Result.id)
            )

            best = float("inf")
            for idx, (time_value, _res_id) in enumerate(fallback, start=1):
                if time_value is None:
                    continue
                best = min(best, time_value)
                data.append((idx, float(best)))

        return data
    finally:
        session.close()
        engine.dispose()


def write_csv(
    csv_path: Path,
    rows: Iterable[Tuple[int, float]],
    experiment: str,
    technique: str,
    rep: int,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "best_time", "experiment", "technique", "rep"])
        for generation, best_time in rows:
            writer.writerow([generation, best_time, experiment, technique, rep])


def run_single_experiment(
    experiment: str,
    config: Dict[str, object],
    technique: str,
    rep: int,
    db_dir: Path,
    results_dir: Path,
) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{experiment}_{technique}_rep{rep}.db"
    if db_path.exists():
        db_path.unlink()

    extra_args = list(config.get("extra_args", []))

    cmd = [
        "uv",
        "run",
        "python",
        str(config["script_path"]),
        *extra_args,
        f"--technique={technique}",
        f"--test-limit={config['test_limit']}",
        f"--database={db_path}",
    ]

    print(f"[{experiment}] {technique} rep {rep}: starting run")
    return_code = run_command(cmd, cwd=Path(config["working_dir"]))
    if return_code != 0:
        print(f"[{experiment}] {technique} rep {rep}: failed with code {return_code}")
        return

    print(f"[{experiment}] {technique} rep {rep}: extracting convergence data")
    convergence = extract_convergence(db_path)
    if not convergence:
        print(f"[{experiment}] {technique} rep {rep}: no convergence data found")
        return

    csv_path = results_dir / f"{experiment}_{technique}_{rep}.csv"
    write_csv(csv_path, convergence, experiment, technique, rep)
    print(f"[{experiment}] {technique} rep {rep}: saved {csv_path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenTuner experiments")
    parser.add_argument(
        "--experiment",
        choices=["rosenbrock", "mmm", "tsp", "gccflags", "all"],
        default="all",
        help="which experiment to run",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="number of repetitions per technique",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    experiments = _build_experiments(repo_root)

    selected = experiments.keys() if args.experiment == "all" else [args.experiment]

    db_dir = repo_root / "opentuner.db"
    results_dir = repo_root / "experiments" / "results"

    for experiment in selected:
        config = experiments[experiment]
        for technique in TECHNIQUES:
            for rep in range(1, args.reps + 1):
                run_single_experiment(
                    experiment=experiment,
                    config=config,
                    technique=technique,
                    rep=rep,
                    db_dir=db_dir,
                    results_dir=results_dir,
                )

    print("All requested runs complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
