# Distilled Feature Fields

This is a simpler and faster demo codebase of [distilled feature fields (DFFs)](https://pfnet-research.github.io/distilled-feature-fields/) (Kobayashi et al. NeurIPS 2022).
Note that this does not contain the comprehensive scripts for all the experiments.

https://user-images.githubusercontent.com/9245278/198859321-9258f101-de76-422a-927a-91fe76d75bbd.mp4

## Example

Setup
```
# assume cuda 11.1
pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 --extra-index-url https://download.pytorch.org/whl/cu111 --no-cache-dir
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.2+cu111.html

pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
git submodule update --init --recursive
cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..
pip install models/csrc/
```

(Download [a sample dataset](https://github.com/pfnet-research/distilled-feature-fields/releases/download/tmp/sample_dataset.zip) or see `With New Scene` section below.)

Train
- `--root_dir` is the dataset of images with poses.
- `--feature_directory` is the dataset of feature maps for distillation. `--feature_dim` matches the dimension of them.
```
python train.py --root_dir sample_dataset --dataset_name colmap --exp_name exp_v1 --downsample 0.25 --num_epochs 4 --batch_size 4096 --scale 4.0 --ray_sampling_strategy same_image --feature_dim 512 --random_bg --feature_directory sample_dataset/rgb_feature_langseg
```

CLIPNeRF-optimize
- `--clipnerf_text rainbow_apple` optimizes the scene to `rainbow apple`
- `--clipnerf_filter_text apple banana vegetable floor` removes rays of `banana`, `vegetable`, and `floor` from optimization, and optimizes rays of `apple` only
- Set `--weight_path` with the checkpoint above.
```
python train.py --root_dir sample_dataset --dataset_name colmap --exp_name exp_v1_clip --downsample 0.25 --num_epochs 1 --batch_size 4096 --scale 4.0 --ray_sampling_strategy same_image --feature_dim 512 --random_bg --clipnerf_text rainbow_apple --clipnerf_filter_text apple banana vegetable floor --weight_path ckpts/colmap/exp_v1/epoch=3_slim.ckpt --accumulate_grad_batches 2
```

Render with Edit
- Modify `--edit_config` or codebase itself for other editings.
- Set `--ckpt_path` with the checkpoint above.
```
python render.py --root_dir sample_dataset --dataset_name colmap --downsample 0.25 --scale 4.0 --ray_sampling_strategy same_image --feature_dim 512 --ckpt_path ckpts/colmap/exp_v1_clip/epoch\=0_slim.ckpt --edit_config query.yaml
# ls ./renderd_*.png
# ffmpeg -framerate 30 -i ./rendered_%03d.png -vcodec libx264 -pix_fmt yuv420p -r 30 video.mp4
```


## With New Scene

### Prepare Posed Images

colmap
```
colmap feature_extractor --ImageReader.camera_model OPENCV --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path sample_dataset/database.db --image_path sample_dataset/images --SiftExtraction.use_gpu=false
colmap exhaustive_matcher --SiftMatching.guided_matching=true --database_path sample_dataset/database.db --SiftMatching.use_gpu=false
mkdir sample_dataset/sparse
colmap mapper --database_path sample_dataset/database.db --image_path sample_dataset/images --output_path sample_dataset/sparse
colmap bundle_adjuster --input_path sample_dataset/sparse/0 --output_path sample_dataset/sparse/0 --BundleAdjustment.refine_principal
_point 1
colmap image_undistorter --image_path sample_dataset/images --input_path sample_dataset/sparse/0 --output_path sample_dataset_undis
--output_type COLMAP
```


### Encode Features by Teacher Network

Setup LSeg
```
cd distilled_feature_field/encoders/lseg_encoder
pip install -r requirements.txt
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
```

Download the LSeg model file `demo_e200.ckpt` from [the Google drive](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing).

Encode and save
```
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ../../sample_dataset_undis/rgb_feature_langseg --test-rgb-dir ../../sample_dataset_undis/images
```
This may produces large feature map files in `--outdir` (100-200MB per file).

Run train.py. If reconstruction fails, change `--scale 4.0` to smaller or larger values, e.g., `--scale 1.0` or `--scale 16.0`.



### Citation

The codebase of NeRF is derived from [ngp_pl](https://github.com/kwea123/ngp_pl/commit/6b2a66928d032967551ab98d5cd84c7ef1b83c3d) (6b2a669, Aug 30 2022) by @kwea123. Thank you.

The codebase of `encoders/lseg_encoder` is derived from [lang-seg](https://github.com/isl-org/lang-seg) by @Boyiliee

The paper bibtex is as follows
```
@inproceedings{kobayashi2022distilledfeaturefields,
  title={Decomposing NeRF for Editing via Feature Field Distillation},
  author={Sosuke Kobayashi and Eiichi Matsumoto and Vincent Sitzmann},
  booktitle={Advances in Neural Information Processing Systems},
  volume = {35},
  url = {https://arxiv.org/pdf/2205.15585.pdf},
  year={2022}
}
```

#### Concurrent work

A concurrent work by [Tschernezki et al.](https://www.robots.ox.ac.uk/~vgg/publications/2022/Tschernezki22/tschernezki22.pdf) also explores feature fields. Please check out [their codebase](https://github.com/dichotomies/N3F).
