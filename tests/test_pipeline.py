import importlib.util
import json
import logging
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import checkpointing
import train
from check_data import audit
from dataset import RiceW3999, local_spacing
from counting_metrics import match_counts
from engine import select_points
from evaluate import load_model
from infer import tile_axis
from losses import PointCriterion
from models import SAHCC, ModelConfig
from models.sah import HungarianMatcherCrowd

torch.set_num_threads(2)


def fixture(root):
    for folder in ('JPEGImages','PointAnnotations','ImageSets'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    names = ['wheat1799__中文 name','ricetip__empty','wheat500__val']
    for i, name in enumerate(names):
        Image.new('RGB',(64,64),(i*60,30,20)).save(root/'JPEGImages'/f'{name}.PNG')
        (root/'PointAnnotations'/f'{name}.txt').write_text('' if i==1 else '20 21\n31 33\n')
    (root/'ImageSets/train.txt').write_text('\ufeff'+names[0]+'\n'+names[1]+'\n')
    (root/'ImageSets/val.txt').write_text(names[2]+'\n')


def args_for(root, tag='run'):
    return train.parser().parse_args(['--data-path',str(root/'data'),'--output-root',str(root),
        '--tag',tag,'--device','cpu','--workers','0','--batch-size','2','--epochs','2',
        '--patch-size','32','--eval-every','1','--scratch'])


class PipelineTests(unittest.TestCase):
    def test_knn_spacing_excludes_self_and_handles_small_sets(self):
        p=np.array([[0,0],[3,0],[0,4]],np.float32)
        np.testing.assert_allclose(local_spacing(p,2),[3.5,4,4.5])
        np.testing.assert_allclose(local_spacing(p,4),local_spacing(p,2))
        self.assertEqual(local_spacing(p[:0]).shape,(0,))
        np.testing.assert_equal(local_spacing(p[:1]),[1])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            local_spacing(np.array([[1,1],[1,1]]))

    def test_crop_preserves_full_image_spacing_and_flip_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root)
            ds=RiceW3999(root,'train',32)
            with patch('dataset.random.uniform',return_value=1.), \
                 patch('dataset.random.randint',side_effect=[0,0]), \
                 patch('dataset.random.random',return_value=0.):
                image,target=ds[0]
            torch.testing.assert_close(target['points'],torch.tensor([[11.,21.]]))
            self.assertAlmostEqual(target['mean_mnn'].item(),float(np.hypot(11,12)),places=5)
            self.assertEqual(tuple(image.shape),(3,32,32))
            self.assertEqual(len(ds[1][1]['points']),0)
            self.assertEqual(audit(root)['train']['images'],2)

    def test_scaling_spacing_and_original_validation_size(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root)
            ds=RiceW3999(root,'train',32)
            with patch('dataset.random.uniform',return_value=1.25), \
                 patch('dataset.random.randint',side_effect=[0,0]), \
                 patch('dataset.random.random',return_value=1.):
                _,target=ds[0]
            self.assertAlmostEqual(target['mean_mnn'][0].item(),float(np.hypot(11,12)*1.25),places=5)
            Image.new('RGB',(513,1023)).save(root/'JPEGImages/wheat500__val.PNG')
            image,target=RiceW3999(root,'val')[0]
            self.assertEqual(tuple(image.shape),(3,1024,528))
            self.assertEqual(target['size'],(1023,513))

    def test_matcher_matches_explicit_cost_and_has_no_grad(self):
        from scipy.optimize import linear_sum_assignment
        torch.manual_seed(13)
        output=dict(pred_logits=torch.randn(2,20,1),anchor_points=torch.rand(2,20,2)*64)
        targets=[dict(points=torch.rand(5,2)*64,mean_mnn=torch.arange(1,6).float()),
                 dict(points=torch.empty(0,2),mean_mnn=torch.empty(0))]
        matcher=HungarianMatcherCrowd()
        result=matcher(output,targets)
        cost=(-output['pred_logits'][0].sigmoid()+torch.cdist(output['anchor_points'][0],
              targets[0]['points'])*(1.2/targets[0]['mean_mnn']).clamp(min=.03)[None,:])
        a,b=linear_sum_assignment(cost.numpy())
        np.testing.assert_equal(result[0][0].numpy(),a)
        np.testing.assert_equal(result[0][1].numpy(),b)
        self.assertEqual(len(result[1][0]),0)
        self.assertEqual(sum(p.numel() for p in matcher.parameters()),0)

    def test_real_model_anchor_geometry_loss_and_gradients(self):
        model=SAHCC(ModelConfig(head_hidden_dim=8,decoder_out_channels=8))
        output=model(torch.randn(2,3,64,64))
        self.assertEqual(tuple(output['pred_points'].shape),(2,256,2))
        torch.testing.assert_close(output['anchor_points'][0,:4],
                                  torch.tensor([[2.,2.],[6.,2.],[2.,6.],[6.,6.]]))
        args=train.parser().parse_args([]);args.cost_point=1.
        targets=[dict(points=torch.tensor([[20.,21.]]),mean_mnn=torch.ones(1),labels=torch.ones(1,dtype=torch.long)),
                 dict(points=torch.empty(0,2),mean_mnn=torch.empty(0),labels=torch.empty(0,dtype=torch.long))]
        losses=PointCriterion(args)(output,targets)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        self.assertGreater(model.head.reg_head.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.backbone.body1[0].weight.grad.abs().sum().item(),0)

    def test_empty_batch_backward_and_capacity_guard(self):
        args=train.parser().parse_args([]);args.cost_point=1.
        criterion=PointCriterion(args)
        output=dict(pred_logits=torch.zeros(1,4,1,requires_grad=True),
                    pred_points=torch.zeros(1,4,2,requires_grad=True),anchor_points=torch.zeros(1,4,2))
        target=dict(points=torch.empty(0,2),mean_mnn=torch.empty(0),labels=torch.empty(0,dtype=torch.long))
        criterion(output,[target])['loss'].backward()
        self.assertIsNotNone(output['pred_points'].grad)
        target=dict(points=torch.ones(5,2),mean_mnn=torch.ones(5),labels=torch.ones(5,dtype=torch.long))
        with self.assertRaisesRegex(ValueError,'More GT'):
            criterion(output,[target])

    def test_sigmoid_threshold_and_padding_filter(self):
        output=dict(pred_logits=torch.tensor([[[2.],[2.],[-2.]]]),
                    pred_points=torch.tensor([[[3.,4.],[70.,3.],[8.,9.]]]))
        points,_=select_points(output,0,(64,64))
        torch.testing.assert_close(points,torch.tensor([[3.,4.]]))

    def test_atomic_failure_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'checkpoint.pth';checkpointing.atomic_save({'v':1},path)
            with patch('checkpointing.torch.save',side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):checkpointing.atomic_save({'v':2},path)
            self.assertEqual(checkpointing.load(path)['v'],1)

    def test_completed_epoch_resume_matches_uninterrupted_training(self):
        small=ModelConfig(head_hidden_dim=8,decoder_out_channels=8)
        def build(config=None,pretrained=None):return SAHCC(small,pretrained)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root/'data')
            with patch('train.SAHCC',side_effect=build):train.main(args_for(root,'continuous'))
            save=checkpointing.save
            def stop_after_epoch(*a,**kw):
                save(*a,**kw)
                if Path(a[0]).name=='checkpoint.pth' and a[5]==0:raise KeyboardInterrupt()
            with patch('train.SAHCC',side_effect=build),patch('train.checkpointing.save',side_effect=stop_after_epoch):
                with self.assertRaises(KeyboardInterrupt):train.main(args_for(root,'resume'))
            with patch('train.SAHCC',side_effect=build):train.main(args_for(root,'resume'))
            first=checkpointing.load(root/'continuous/output/checkpoint.pth')
            second=checkpointing.load(root/'resume/output/checkpoint.pth')
            for key,value in first['model'].items():
                torch.testing.assert_close(value,second['model'][key],rtol=0,atol=0)
            self.assertEqual(first['best_mae'],second['best_mae'])
            self.assertEqual(first['lr_scheduler'],second['lr_scheduler'])
            model,_=load_model(root/'resume/output/checkpoint.pth',torch.device('cpu'))
            self.assertEqual(model.config,small)
            report=json.loads((root/'resume/output/best_metrics.json').read_text())
            self.assertIn('rmse',report['metrics'])
            self.assertTrue((root/'resume/output/evaluation/epoch_0002/per_source.csv').is_file())
            changed=args_for(root,'resume');changed.eval_every=5
            with patch('train.SAHCC',side_effect=build):train.main(changed)
            changed=args_for(root,'resume');changed.neighbors=3
            with patch('train.SAHCC',side_effect=build):
                with self.assertRaisesRegex(ValueError,'neighbors'):train.main(changed)

    def test_tiles_cover_once_without_radius_merging(self):
        intervals=tile_axis(3072,1024,64)
        for point in range(3072):
            self.assertEqual(sum(lo<=point<hi for _,_,lo,hi in intervals),1)

    def test_baseline_ignores_spacing_and_localization_is_one_to_one(self):
        args=train.parser().parse_args(['--matcher','baseline']);args.cost_point=.05
        criterion=PointCriterion(args)
        output=dict(pred_logits=torch.tensor([[[1.],[2.],[0.]]]),
                    pred_points=torch.tensor([[[1.,1.],[5.,5.],[10.,10.]]]),
                    anchor_points=torch.tensor([[[1.,1.],[5.,5.],[10.,10.]]]))
        t=dict(points=torch.tensor([[3.,3.]]),mean_mnn=torch.ones(1),labels=torch.ones(1,dtype=torch.long))
        first=criterion(output,[t])['loss']
        t['mean_mnn']=torch.ones(1)*100
        torch.testing.assert_close(first,criterion(output,[t])['loss'])
        self.assertEqual(match_counts([[0,0],[0,1]],[[0,0]],4),(1,1,0))

    def test_validation_at_five_and_final_epoch(self):
        def build(config=None,pretrained=None):
            return SAHCC(ModelConfig(head_hidden_dim=8,decoder_out_channels=8),pretrained)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root/'data');args=args_for(root)
            args.epochs=6;args.eval_every=5
            with patch('train.SAHCC',side_effect=build):train.main(args)
            folders=sorted(p.name for p in (root/'run/output/evaluation').iterdir())
            self.assertEqual(folders,['epoch_0005','epoch_0006'])


if __name__=='__main__':unittest.main()
