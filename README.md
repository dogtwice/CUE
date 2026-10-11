# CUE

**CUE — Conflict and Uncertainty-aware Evidential graph-level clustering.**

CUE models each augmented graph view as a Dirichlet-based evidential opinion,
suppresses unreliable views via view-level uncertainty, resolves inter-view
evidential conflict through uncertainty-driven fusion (UDF, PCR6), and feeds
the resulting sample-level uncertainty back into contrastive learning through
an adaptive temperature. This repository ships the code needed to train and
evaluate the model on graph-level clustering benchmarks using **ACC / NMI / ARI**.

## Environment

Python ≥ 3.10 with PyTorch and PyTorch Geometric (CUDA build recommended).

```bash
conda env create -f environment.yml
conda activate stcmc-ur
# or, with an existing PyTorch + PyG environment:
pip install -r requirements.txt
```

## Quick start

The MUTAG dataset is already included under `data/raw/MUTAG/raw/`, so no
download is needed. The MUTAG experiment configuration is defined in
`src/config/experiments/mutag.py`.

```bash
python -m src.models.train -c mutag
```

On Windows, pin the data-loader worker count to 0 for stability:

```bash
CUE_NUM_WORKERS=0 python -m src.models.train -c mutag
```

