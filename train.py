import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
import os
import time
os.environ["TORCH_HOME"] = "/tmp/torch/"

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")

import clip
import yaml
from clip_utils import CLIPEditor


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        if hparams.feature_directory is not None:
            assert hparams.feature_dim is not None, "set feature_dim for using feature field"
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act,
                         feature_out_dim=hparams.feature_dim)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

        # edit
        if hparams.edit_config is not None or hparams.clipnerf_text is not None:
            self.clip_editor = CLIPEditor()
        """
        if hparams.edit_config is not None:
            with open(hparams.edit_config, 'r') as f:
                edit_config = yaml.safe_load(f)

            # setup query
            self.model.positive_ids = edit_config.positive_ids
            self.model.score_threshold = edit_config.score_threshold
            if edit_config.query.query_type == "text":
                #clip_model, _ = clip.load("ViT-B/32", device="cuda")
                #tokenized_texts = clip.tokenize(edit_config.query.texts)
                #with torch.no_grad():
                #    text_features = clip_model.encode_text(tokenized_texts)
                #    text_features /= text_features.norm(dim=-1, keepdim=True)
                self.model.query_features = self.clip_editor.encode_text(edit_config.query.texts.replace('_', ' '))
            else:
                raise NotImplementedError

            # setup editing
            self.edit_dict = {}
            for op in edit_config.operations:
                if op.edit_type == "deletion":
                    self.edit_dict["deletion"] = True
                elif op.edit_type == "color_func":
                    self.edit_dict["color_func"] = eval(op.func_str)
                else:
                    raise NotImplementedError
        else:
            self.model.query_features = None
        """

        # clipnerf
        if hparams.clipnerf_text is not None:
            self.clip_editor.text_features = self.clip_editor.encode_text([hparams.clipnerf_text.replace('_', ' ')])
        if hparams.clipnerf_filter_text is not None:
            self.clip_editor.text_filter_features = self.clip_editor.encode_text(
                [t.replace('_', ' ') for t in hparams.clipnerf_filter_text])
            print([t.replace('_', ' ') for t in hparams.clipnerf_filter_text])
            print(self.clip_editor.text_filter_features @ self.clip_editor.text_filter_features.T)


    def calculate_clip_loss(self, results, batch):
        patch_size = self.train_dataset.patch_size
        rendered_patch = results['rgb'].reshape(1, patch_size, patch_size, 3).permute(0, 3, 1, 2)  # (nhw,c) -> (n,c,h,w)
        gt_patch = batch['rgb'].reshape(1, patch_size, patch_size, 3).permute(0, 3, 1, 2)  # (n,c,h,w)

        # detach pixels of non-queried regions
        if self.clip_editor.text_filter_features is not None:
            rendered_features = results['feature']
            scores = self.model.calculate_selection_score(rendered_features, query_features=self.clip_editor.text_filter_features)
            score_patch = scores.reshape(1, patch_size, patch_size, 1).permute(0, 3, 1, 2).detach()
            rendered_patch = rendered_patch * score_patch + rendered_patch.detach() * (1 - score_patch)

        # make rendered patch (with/without augmentations) similar to target text via clip
        sample_N_aug = 5  # N random augmentations
        clip_emb = self.clip_editor.encode_image(rendered_patch, preprocess=True, stochastic=sample_N_aug)  # (N_aug, dim)
        clip_loss = 1.0 - (self.clip_editor.text_features.float()[None] * clip_emb).sum(dim=-1)
        losses = {'cliploss': clip_loss.mean()}

        # render for debug
        render_for_debug = False
        if render_for_debug:
            rgb_pred = (rendered_patch.detach().permute(0, 2, 3, 1)[0].cpu().numpy()*255).astype(np.uint8)  # (h,w,c)
            imageio.imsave('tmpdebug_{}__{}.png'.format(time.time(), clip_loss.mean()), rgb_pred)
            if self.clip_editor.text_filter_features is not None:
                score_pred = (score_patch.detach().permute(0, 2, 3, 1)[0].cpu().numpy()*255).astype(np.uint8)[:, :, 0]  # (h,w)
                imageio.imsave('tmpdebug_{}__{}_score.png'.format(time.time(), clip_loss.mean()), score_pred)                
                feat_patch = rendered_features.reshape(1, patch_size, patch_size, -1)[0, :, :, :3]
                feat_patch = (feat_patch - feat_patch.min()) / (feat_patch.max() - feat_patch.min())
                feat_pred = (feat_patch.detach().cpu().numpy()*255).astype(np.uint8)  # (h,w,c)
                imageio.imsave('tmpdebug_{}__{}_feat.png'.format(time.time(), clip_loss.mean()), feat_pred)

        # preserve original scene in non-queried regions as possible
        preserve_original = True
        if preserve_original:
            if self.clip_editor.text_filter_features is not None:
                rendered_patch = results['rgb'].reshape(1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
                rendered_patch = rendered_patch * (1 - score_patch) \
                                 + rendered_patch.detach() * score_patch
            rgb_loss_gt = ((rendered_patch - gt_patch) ** 2)
            losses['rgb_loss_gt'] = rgb_loss_gt.mean() * 10.0
            # clip_emb_gt = self.clip_editor.encode_image(gt_patch, preprocess=True, stochastic=sample_N_aug)[0].detach()
            # clip_loss_gt = 1.0 - (clip_emb * clip_emb_gt).sum(dim=-1)
            # losses['cliploss_gt'] = clip_loss_gt.mean()
        return losses

    def forward(self, batch, split, detach_geometry=False):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg,
                  'detach_geometry': detach_geometry}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']
        if split=='test':
            kwargs['render_feature'] = True

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        if self.hparams.clipnerf_text is not None:
            kwargs['len_per_epoch'] = 200  # often sufficient
        self.train_dataset = dataset(split=self.hparams.split,
                                     load_features=hparams.feature_directory is not None,
                                     feature_directory=hparams.feature_directory,
                                     **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy
        if self.hparams.clipnerf_text is not None:
            self.train_dataset.patch_size = self.hparams.clipnerf_patch_size

        self.test_dataset = dataset(split='test', **kwargs)

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT'] and not n.startswith('val_lpips'):
                net_params += [p]
                print(n, p.shape, 'to be optimized')

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=8,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=4,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')

        if self.hparams.clipnerf_text is not None:
            # TODO: generate random poses + training poses
            results = self(batch, split='train', detach_geometry=True)
            loss_d = self.calculate_clip_loss(results, batch)
        else:
            results = self(batch, split='train')
            loss_d = self.loss(results, batch)

            if self.global_step % (2*self.update_interval) == 0 and self.hparams.clipnerf_text is None:
                # regularization for cleaning
                loss_d['density_mean'] = self.model.sample_density(
                    0.01*MAX_SAMPLES/3**0.5, warmup=self.global_step<self.warmup_steps).mean() * 1e-4

        # feature loss
        if 'feature' in results and self.hparams.clipnerf_text is None:
            loss_d['feature'] = ((results['feature'] - batch['feature']) ** 2).sum(-1).mean() * 1e-2
            self.log('train/loss_f', loss_d['feature'])

        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v.mean())
        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        with torch.no_grad():
            results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

            # visualize PCA feature
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
                if not hasattr(self, 'proj_V'):
                    U, S, V = torch.pca_lowrank(
                        (results['feature'] - results['feature'].mean(0)[None]).float(),
                        niter=5)
                    self.proj_V = V[:, :3].float()
                    lowrank = torch.matmul(results['feature'].float(), self.proj_V)
                    self.lowrank_sub = lowrank.min(0, keepdim=True)[0]
                    self.lowrank_div = lowrank.max(0, keepdim=True)[0] - lowrank.min(0, keepdim=True)[0]
                else:
                    lowrank = torch.matmul(results['feature'].float(), self.proj_V)
                lowrank = ((lowrank - lowrank.min(0, keepdim=True)[0]) / (lowrank.max(0, keepdim=True)[0] - lowrank.min(0, keepdim=True)[0])).clip(0, 1)

                visfeat = rearrange(lowrank.cpu().numpy(), '(h w) c -> h w c', h=h)
                visfeat = (visfeat*255).astype(np.uint8)
                imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_f.png'), visfeat)

            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      # log_every_n_steps=5,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      accumulate_grad_batches=hparams.accumulate_grad_batches,
                      # amp_backend="apex",
                      # amp_level="O1",
                      precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')
        print(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
