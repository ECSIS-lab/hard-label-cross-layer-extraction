# CrossLayer Extraction — Proof of Concept

This repository provides a proof-of-concept implementation of CrossLayer Extraction from the paper **[Is the Hard-Label Cryptanalytic Model Extraction Really Polynomial?](https://doi.org/10.1007/978-3-032-35415-0_3)** by Akira Ito, Takayuki Miura, and Yosuke Todo (CRYPTO 2026).

The code supports experiments on cross-layer consistency checking and span recovery. It is intended to validate these components under controlled assumptions, and is not a complete end-to-end hard-label model extraction implementation.

## Scope and assumptions

This code is a proof of concept that uses model internals to evaluate CrossLayer Extraction; it is not designed to carry out an attack in a real black-box setting.

For dual-point extraction and conventional signature-recovery procedures, please refer to Carlini et al., **[Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting](https://doi.org/10.1007/978-3-031-91107-1_13)** (EUROCRYPT 2025), and their **[original implementation](https://github.com/Jchavezsaab/hard-label-dnn-extraction)**.

## Installation

Python 3.10 or newer is recommended. The code requires PyTorch, NumPy, and SciPy.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Repository contents and models

`consistent.py` contains the experimental routines and command-line interface. Each model directory contains a `state_dict.pt` checkpoint and its `training_metadata.json`.

| Hidden width | Model directory | Training epochs |
| --- | --- | --- |
| 64 | `68db5deee325620a018a1ba1/` | 0 (untrained) |
| 128 | `68db5dfb8a418b4f3ec1e70c/` | 0 (untrained) |
| 256 | `68d77c2b1e8717de2410cf3d/` | 30 |

The supplied models use an MNIST input/output configuration: 784 inputs, nine ReLU hidden layers, and 10 outputs. The training metadata counts all 10 linear layers as `depth=10`; the script uses `depth=9` for the hidden-layer count. Epoch counts above are taken from the supplied metadata.

## Usage

Run commands from the repository root, which contains `consistent.py`, so the relative model paths resolve correctly. Keep the model directories in their supplied locations.

```bash
python consistent.py --hidden-dim 64 --check-intersection-space heuristic --run consistent-right
```

Running without arguments displays help without starting an experiment:

```bash
python consistent.py
```

### Experiment modes

| `--run` value | Experiment |
| --- | --- |
| `consistent-right` | Cross-layer consistency checking using intersection spaces associated with the same target neuron (`right_set=True`). |
| `consistent-wrong` | Cross-layer consistency checking with the first intersection space associated with a different neuron (`right_set=False`). |
| `span-recovery` | Cross-layer span recovery. |

For example, to run the other two modes:

```bash
python consistent.py --hidden-dim 64 --check-intersection-space heuristic --run consistent-wrong
python consistent.py --hidden-dim 64 --check-intersection-space heuristic --run span-recovery
```

### Options

| Option | Choices | Default |
| --- | --- | --- |
| `--hidden-dim` | `64`, `128`, `256` | `64` |
| `--check-intersection-space` | `heuristic`, `imaginary` | `heuristic` |
| `--run` | `span-recovery`, `consistent-right`, `consistent-wrong` | `consistent-right` |

`--hidden-dim` selects the supplied model by its hidden-layer width.

Both intersection-space modes use model internals:

- `heuristic` runs a routine to check whether the target intersection space exists. This adds computation.
- `imaginary` skips that routine and does not verify whether the intersection space is reachable from the input space.

In our experiments, these modes led to broadly similar conclusions, with runtime being the main practical difference.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{ito2026crosslayer,
  author    = {Akira Ito and Takayuki Miura and Yosuke Todo},
  title     = {Is the {Hard-Label} Cryptanalytic Model Extraction Really Polynomial?},
  booktitle = {Advances in Cryptology -- CRYPTO 2026},
  editor    = {Nadia Heninger and Mike Rosulek},
  series    = {Lecture Notes in Computer Science},
  volume    = {16806},
  pages     = {67--98},
  year      = {2026},
  publisher = {Springer},
  address   = {Cham},
  doi       = {10.1007/978-3-032-35415-0_3},
  url       = {https://doi.org/10.1007/978-3-032-35415-0_3}
}
```
