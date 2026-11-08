"""Official SAH/baseline matching and sigmoid BCE + point regression."""
from types import SimpleNamespace

import torch
from torch import nn

from models.matcher import build_matcher_crowd
from models.criterion import CrowdCriterion


class PointCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        config = SimpleNamespace(
            matcher_impl='sah' if args.matcher == 'sah' else 'matcher',
            set_cost_class=args.cost_class, set_cost_point=args.cost_point,
            matcher_point_source=args.point_source,
            matcher_k=args.kappa, matcher_alpha_min=args.alpha_min)
        self.criterion = CrowdCriterion(build_matcher_crowd(config), args.point_weight)

    def forward(self, outputs, targets):
        # Assignment and losses must remain FP32 under optional AMP.
        values = {k: v.float() for k, v in outputs.items()}
        for i, target in enumerate(targets):
            if len(target['points']) > len(values['pred_points'][i]):
                raise ValueError('More GT points than candidates; increase candidate density')
            if not torch.isfinite(target['mean_mnn']).all() or (target['mean_mnn'] <= 0).any():
                raise ValueError('Invalid local spacing')
        with torch.autocast(values['pred_logits'].device.type, enabled=False):
            losses = self.criterion(values, targets)
            weighted = {key: value * self.criterion.weight_dict[key] for key, value in losses.items()}
            return dict(loss=sum(weighted.values()), **weighted)
