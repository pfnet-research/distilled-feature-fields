from torch.utils.data import Dataset
import numpy as np
import torch


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0, len_per_epoch=1000):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.len_per_epoch = len_per_epoch
        self.patch_size = None  # 128
        self.patch_coverage = 0.9

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return self.len_per_epoch
        return len(self.poses)

    def sample_patch(self, h, w):
        skip = int((min(h, w) * self.patch_coverage) / self.patch_size)
        patch_w_skip = self.patch_size * skip
        patch_h_skip = self.patch_size * skip

        left = torch.randint(0, w - patch_w_skip - 1, (1,))[0]
        left_to_right = torch.arange(left, left + patch_w_skip, skip)
        top = torch.randint(0, h - patch_h_skip - 1, (1,))[0]
        top_to_bottom = torch.arange(top, top + patch_h_skip, skip)

        index_hw = (top_to_bottom * w)[:, None] + left_to_right[None, :]
        return index_hw.reshape(-1)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            # randomly select pixels
            if self.patch_size is None:
                pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            else:
                pix_idxs = self.sample_patch(self.img_wh[1], self.img_wh[0])

            rays = self.rays[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3]}

            if hasattr(self, 'features') and len(self.features):
                if self.ray_sampling_strategy == 'all_images':
                    # TODO
                    raise NotImplementedError
                elif self.ray_sampling_strategy == 'same_image':
                    feature_map = self.features[img_idxs][None].float()  # chw->1chw
                    u = (pix_idxs % self.img_wh[0] / self.img_wh[0]) * 2 - 1
                    v = (pix_idxs // self.img_wh[0] / self.img_wh[1]) * 2 - 1
                    with torch.no_grad():
                        sampler = torch.tensor(np.stack([u, v], axis=-1)[None, None]).float()  # N2->11N2
                        # TODO: sparse supervision
                        feats = torch.nn.functional.grid_sample(feature_map, sampler, mode='bilinear', align_corners=True)  # 1c1N
                        feats = feats[0, :, 0].T  # 1c1N->cN->Nc
                    sample['feature'] = feats

            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample
