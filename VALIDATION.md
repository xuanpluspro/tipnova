# Validation record

TipNova migration checks: Python compilation, shell syntax, and nine dataset-audit tests passed. No GPU training or accuracy measurement was run.

The following is inherited from the SAH baseline and does not validate TipNova DDG.

CPU checks completed in the development environment with Python 3.12 and
PyTorch 2.15.0+cpu; CUDA was unavailable.

- `python -m unittest discover -s tests -v`: **12 passed**.
  Includes exact kNN distances, full-image spacing through crop/scale/flip,
  Chinese/space IDs, empty labels, mixed original dimensions and padding,
  explicit SAH cost-matrix assignment, foreground sigmoid filtering,
  point-capacity validation, background-only gradients, baseline spacing
  independence, one-to-one metrics, atomic interrupted writes, tile ownership,
  full epoch-resume model-tensor equality, configuration guards and validation
  at epoch 5 plus final epoch 6.
- Copied official predictor and SAH matcher from the pinned upstream were
  independently imported and compared with this adapter. With equal weights,
  outputs are bit-exact and assignments identical on a synthetic 64px input.
  Small head dimensions were used for this parity check; architecture code is
  otherwise unchanged.
- Default full-size model (head/decoder 256 channels), 512x512 FP32 synthetic
  input: forward, assignment with 100 synthetic points, finite loss and backward
  passed. Output shape `[1, 16384, 2]`; **17,373,772 parameters**.
- Independent evaluate, tiled infer and benchmark CLI commands passed using a
  synthetic 64px fixture and small-head checkpoint. JSON/CSV, xy point files and
  annotated JPEG exports were produced. This was not an accuracy benchmark.
- Python compile and training shell syntax checks passed.

Not verified: actual RiceW3999 files, supplied ImageNet weight-file loading,
GPUFree CUDA/AMP behavior, batch-32 memory, real training throughput or accuracy.
No claim of paper benchmark reproduction, improved leaf-tip accuracy, or measured
GPU capacity is made. Resume equality was tested on CPU with workers=0 and a
small head; changing batch/workers/hardware can alter training trajectories.
