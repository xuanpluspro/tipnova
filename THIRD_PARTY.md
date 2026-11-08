# Attribution

- SAHCC authors, MIT: https://github.com/kaijiang77/SAHCC,
  commit `aaca0838a0749d200661ef7070183aa3131e1681`.
  Predictor, matching algorithms and loss adapted in `models/`.
  Original copyright and license are retained in LICENSE.
- Engineering utilities adapted from the user's
  https://github.com/xuanpluspro/apgcc-leaf-counting at
  `f6564f2c2e7a8f074b812a6218e172bef2564193`:
  data checks, offline VGG feature loader, evaluation protocol, completed-epoch
  checkpointing, train/evaluate/infer/benchmark CLI and progress infrastructure.
  Its MIT notice is retained at licenses/APGCC-MIT.txt. No APGCC IFI or auxiliary
  point decoder is included in this predictor.
- VGG16-BN pretrained weights are not redistributed. Supply a trusted local
  torchvision-format ImageNet state dict and comply with its source terms.

Please cite the SAHCC paper *Spacing-Aware Hungarian Matching for Crowd Counting*
when using the SAH algorithm. This is a leaf-tip data adapter, not the official
benchmark release or a newly claimed original SAH algorithm.

TipNova derives from https://github.com/xuanpluspro/sah-leaf-counting at `1d03815c784dbe7b632d0ab620f053b0c446a9d6`. It adds DDG to the adapted SAHCC predictor.

Dataset references: [FoMo4Wheat Rice Leaf](https://arxiv.org/abs/2509.06907), [Li et al.](https://doi.org/10.34133/plantphenomics.0041), [Yang et al.](https://doi.org/10.1016/j.compag.2025.111053).
