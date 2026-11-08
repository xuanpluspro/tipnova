# Reproduction scope

## Pinned upstream

SAHCC: https://github.com/kaijiang77/SAHCC at
`aaca0838a0749d200661ef7070183aa3131e1681` (MIT).

`models/sahcc.py`, `sah.py`, `matcher.py`, `criterion.py` are adapted from its
predictor, matchers and loss. Predictor equations, VGG stage slicing, sigmoid
classification, anchor geometry, offset scaling and loss normalization are retained.
The anchor grid cache is bounded to eight spatial grids. Imports use this package.
SAH distance weights remain `max(kappa/mean_mnn, alpha_min)`; there is no learned
weight, upper cap, radius cutoff or test-time SAH module.

The backbone is the same feature-only VGG16-BN staging as upstream, initialized
from local torchvision-format ImageNet weights. We omit the unused classification
layers and support the user's `.pth` filename instead of upstream's `.pt` default.
We do not download weights or load torch from requirements.

## Deliberate data/training adaptations

- VOC-like RiceW3999 xy TXT, full-line Unicode IDs instead of SHHA npz.
- Neighbors default to **4 other points**, following the paper. The upstream
  prepare_shha helper defaults to **3**; this discrepancy is explicit and can be
  tested with `--neighbors 3`.
- scipy cKDTree computes exact Euclidean kNN distances without an NxN matrix.
  Original-image neighborhoods are preserved across crops; exact rounded resize
  geometry is used before selecting visible points. Singleton uses upstream 1 px
  spacing (scaled on augmentation); duplicates fail explicitly.
- One online 512 patch per image, batch 32, 50 epochs vs upstream SHHA defaults:
  four 128 patches per source, source batch 8, 2000 epochs. These are different
  optimization schedules and do not reproduce reported SHHA accuracy.
- PIL RGB resize/flip instead of upstream normalized-tensor interpolation;
  boundary-safe `W-1-x` horizontal flip. No photometric augmentation is added.
- Official AdamW 1e-4/1e-5, wd 1e-4, clipping .1 and code's StepLR 3500/gamma .1.
  No LR decay occurs in a 50 epoch run. Batch increase does not automatically
  scale learning rate. Optional AMP is an engineering option, not paper parity.
- Plain single-GPU PyTorch replaces Lightning/Hydra; restartable epoch checkpoints,
  progress bars, evaluation exports and inference are local engineering utilities.
- Validation batch 1, native dimensions padded to 16. Predictions outside the
  original image are discarded. No test-time resizing or radius NMS.
- Checkpoint and CSV schemas are local and not drop-in compatible with official
  Lightning checkpoints. Load only trusted checkpoints (full resume uses pickle).

## Controlled comparisons

Use a new tag per configuration. Fixed network, seed, data, patches and optimizer;
compare baseline+anchor vs SAH+anchor, and predicted-point vs anchor references.
Baseline's default cost_point=.05 vs SAH's 1 follows upstream; sweep baseline
weights before attributing improvement solely to spacing. Both use the same
sigmoid probabilities in classification costs and same BCE/regression criterion.
Set at least three seeds when making scientific claims; 50 epochs is an initial
budget, not proof of convergence.

Spacing has no upper cap in the official algorithm. Incorrect/near-duplicate point
annotations can amplify weights; inspect diagnostics and labels instead of silently
adding a cap and calling it the original method. Severe annotation corruption is
an explicit limitation in the SAH paper.

Native-size F1@4/8/16px uses the shared PET/P2R/APGCC evaluation utility; it is not
the variable head-size NWPU protocol. Timing and parameter counts must be measured
on the user's actual environment. CPU synthetic smoke tests establish software
behavior, not RiceW3999 accuracy, CUDA stability, or L40 batch-32 capacity.

## TipNova delta

Training enables DDG before C4/C5 fusion by default. Use `--no-directional-detail` for the same-code SAH baseline. Prior parity results apply only to the disabled configuration. DDG adds 2,817 parameters at 256 channels.
