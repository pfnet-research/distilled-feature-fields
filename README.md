# Distilled Feature Fields


### Setup

Maybe, using conda is easy.
```
conda create -n dff python=3.8 -y && conda activate dff
```

Install torches.
```
# cu113
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# cu111
pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 --extra-index-url https://download.pytorch.org/whl/cu111 --no-cache-dir
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.2+cu111.html
```

Install others.
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd .. && pip install -r requirements.txt && pip install --upgrade pip && pip install models/csrc/ && pip install setuptools==59.5.0 && pip install pyquaternion plyfile && pip install pytransform3d
```


### Image Encoder

#### LSeg

```
cd distilled_feature_field/encoders/lseg_encoder
pip install -r requirements.txt
```

```
Download the model file from https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing
```

```
python -u encode_images.py --backbone clip_vitl16_384 --data-path aaa --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ./rgb_feature_langseg_075_1_125_175 --test-rgb-dir mynerfs_colmap/veges_dense_colmap/images_4 --dataset ignore
```


### DFF

```
python train.py --root_dir mynerfs_colmap/veges_dense_colmap --dataset_name colmap --exp_name v3_veges_dense_s8_f_max128_e5 --downsample $DOWNSAMPLE --num_epochs 5 --batch_size 4096 --scale 8.0 --eval_lpips --feature_directory rgb_feature_langseg_075_1_125_175_veges_dense --ray_sampling_strategy same_image --feature_dim 512
```


