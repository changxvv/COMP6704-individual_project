# Derivative-Free Optimization for Compiler Phase-Ordering

Comparative study of derivative-free optimization techniques for compiler flag selection and phase-ordering using OpenTuner.

## Overview

This project evaluates five optimization techniques on the compiler flag tuning problem (GCCFlags) with supporting benchmarks:

| Technique | Type |
|-----------|------|
| PureRandom | Baseline |
| DifferentialEvolution | Evolutionary |
| RandomNelderMead | Simplex |
| PSO_GA_Bandit | Hybrid |
| AUCBanditMetaTechniqueA | Adaptive meta-technique |

## Quick Start

```bash
# Install dependencies
uv pip install -r opentuner/requirements.txt -r opentuner/optional-requirements.txt

# Run all experiments (~30 min)
bash experiments/run_all.sh

# Generate figures only
uv run python experiments/plot_results.py
```

## Project Structure

```
├── experiments/
│   ├── run_experiments.py   # Main experiment runner
│   ├── plot_results.py      # Figure generation
│   ├── results/             # CSV outputs
│   └── figures/             # PNG figures
├── opentuner/               # OpenTuner framework (submodule)
└── autotuning_literature_research.md  # Literature survey
```

## Benchmarks

- **GCCFlags** (primary): Compiler flag optimization (~10^806 configurations)
- **Rosenbrock**: Continuous optimization validation
- **MMM**: Integer parameter tuning (matrix multiplication)
- **TSP**: Permutation optimization (15-city tour)

## Key Findings

1. Bandit meta-techniques provide robust cross-domain performance
2. Nelder-Mead excels on continuous spaces but fails on permutations
3. No single technique dominates all problem types
4. Parameter type strongly influences technique effectiveness

## Requirements

- Python 3.8+
- GCC (for GCCFlags experiment)
