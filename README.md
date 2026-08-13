# Life Finds a Way — simulation code

Simulation code accompanying the manuscript *Life Finds a Way* (submitted to the *Journal of
Theoretical Biology*, JTB-D-25-00796).

Preprint: https://arxiv.org/abs/2507.13253

## What the model does

An agent-based model of a growing directed network in which nodes acquire directed `help` and
`harm` relations to one another. Each node carries an output trait and a binding tolerance; a
relation forms between two nodes when their traits fall within `bind_range` of one another. The
sign of that relation (help or harm) is set by the harm-to-help ratio `rho`.

The quantity of interest is whether the population sustains **bounded or unbounded growth** as
`rho` varies, and whether a **strongly connected component (SCC)** — a set of nodes that
mutually sustain one another — emerges and persists.

## Contents

| File | Purpose |
|---|---|
| `LFAWNoMatrices.py` | Complete model, batch runner, and figure generation |

The module is self-contained. `simulate_network()` runs a single realization,
`run_batches()` averages over replicates, and the plotting functions produce the diagnostic
panels and the growth-rate-versus-`rho` figure reported in the manuscript.

## Reproducing the manuscript figures

```bash
pip install -r requirements.txt
python LFAWNoMatrices.py
```

`main()` performs the full sweep reported in the paper:

| Parameter | Value | Meaning |
|---|---|---|
| `help_hurt` (`rho`) | `linspace(0, 1, 21)` | Harm-to-help ratio, 21 points |
| `num_steps` | `10000` | Simulation steps per run |
| `num_batches` | `3` | Replicates averaged per `rho` |
| `bind_range` | `0.003` | Trait tolerance for relation formation |
| `lifespan` | `100` | Node lifespan in steps |
| `seed` | `42` | Base RNG seed |

Output written to the working directory:

- `rho_sweep_lifespan100_bind0.003_rho<R>_steps10000.png` — per-`rho` diagnostic panel
- `late_growth_vs_rho_full_sweep.png` — late-time growth rate `g` against `rho`, the
  bounded/unbounded transition

The sweep is the expensive step. Runtime scales with `num_steps` × `num_batches` × 21; reduce
`num_batches` or `num_steps` in `main()` for a faster smoke test.

Seeding is deterministic given `seed`, so a rerun at identical parameters reproduces the
reported figures.

## Requirements

Python 3.9 or later. See `requirements.txt`.

`leidenalg` and `python-igraph` supply the community-detection routines; `cdlib` wraps them.
On platforms without prebuilt wheels these require a C toolchain.

## Citation

If you use this code, please cite the manuscript. A full citation will be added here on
acceptance.

## License

MIT — see `LICENSE`.
