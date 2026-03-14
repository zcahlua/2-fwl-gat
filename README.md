# 2-fwl-gat

This repository now contains **two independent codepaths**:

1. **Original TensorFlow 1.x Cora GAT node-classification path** (preserved under `legacy/`).
2. **New PyTorch Geometric QM9 graph-regression path** (`qm9_local2fwl/`) implementing a **local / sparsified 2-FWL-style GAT**.

## Preserved legacy Cora path (unchanged)

The original implementation is still present and runnable:

- `legacy/execute_cora.py`
- `legacy/models/`
- `legacy/utils/`

## New QM9 Local 2-FWL-style GAT path

The new path is intentionally separate from legacy TensorFlow code and targets **graph-level regression** on QM9.

### Install

```bash
pip install -r requirements.txt
pip install -r requirements-qm9.txt
```

### Run training

```bash
python -m qm9_local2fwl.train --target 0 --epochs 5 --batch-size 16 --subset 512
```

You can switch targets with `--target <index>` and use full data by omitting `--subset`.

## Local 2-FWL-style operator (honest scope)

This is **not full global 2-FWL**. It is a local/sparsified variant:

- Build a sparse active set of unordered pair states `{a,b}` using bonded pairs + required triplet-induced pairs.
- For each source atom `u`, select top-2 nearest neighbors `(v,w)` from 3D coordinates.
- Form source-centered triplet `(v,u,w)` and gather pair states:
  - `h_vu`
  - `h_uw`
  - `h_vw`
- Compute triplet messages `psi(h_vu, h_uw, h_vw, g_vuw)` using geometry (`d_uv,d_uw,d_vw,angle_vuw`).
- Scatter-add triplet messages to target pair `{v,w}` and update pair states.
- Fuse pair states back to **both endpoints** of each pair.
- Apply bond-graph `GATConv` node refinement.
- Global mean pool and regress one scalar target.

## New file layout

- `qm9_local2fwl/__init__.py`
- `qm9_local2fwl/data.py`
- `qm9_local2fwl/model.py`
- `qm9_local2fwl/train.py`
- `qm9_local2fwl/utils.py`
- `requirements-qm9.txt`
- `tests/test_qm9_smoke.py`

## Known limitations

- This is a local 2-FWL-style approximation, not exact global 2-FWL.
- Top-2 neighbor triplets are geometry-driven and may miss longer-range interactions.
- Per-batch construction of pair/triplet structures is straightforward but not heavily optimized yet.
