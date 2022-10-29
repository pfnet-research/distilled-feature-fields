import copy
import glob
import os
import itertools
import functools
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as torch_transforms
import encoding.datasets as enc_ds
from PIL import Image

encoding_datasets = {
    x: functools.partial(enc_ds.get_dataset, x)
    for x in ["coco", "ade20k", "pascal_voc", "pascal_aug", "pcontext", "citys"]
}


class FolderLoader(enc_ds.ADE20KSegmentation):#(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = get_folder_images(root)
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n"))
        # self.num_class = 150  # ADE20k

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)


def get_folder_images(img_folder):
    img_paths = []
    # sorted(glob.glob(self.rgb_dir + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
    # for filename in sorted(os.listdir(img_folder)):
    # rgb_885.png
    # for filename in sorted(glob.glob(img_folder.rstrip("/") + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4])):
    #if "colmap" in img_folder or "llff" in img_folder:
    glist = list(glob.glob(img_folder.rstrip("/") + '/*.png')) + list(glob.glob(img_folder.rstrip("/") + '/*.jpg'))
    glist = sorted(glist)
    #for rep
    #else:
    #    glist = glob.glob(img_folder.rstrip("/") + '/rgb*.png')
    #    glist = sorted(glist, key=lambda file_name: int(file_name.split("_")[-1][:-4]))
    for filename in glist:
        #if filename.endswith(".jpg") or filename.endswith(".png"):
        imgpath = os.path.join(img_folder, filename)
        img_paths.append(imgpath)
        print(imgpath)
    return img_paths


def get_dataset(name, **kwargs):
    if name in encoding_datasets:
        return encoding_datasets[name.lower()](**kwargs)
    if os.path.isdir(name):
        print("load", name, "as image directroy for FolderLoader")
        return FolderLoader(name, transform=kwargs["transform"])
    assert False, f"dataset {name} not found"


def get_original_dataset(name, **kwargs):
    if os.path.isdir(name):
        print("load", name, "as image directroy for FolderLoader")
        return FolderLoader(name, transform=kwargs["transform"])
    assert False, f"dataset {name} not found"


def get_available_datasets():
    return list(encoding_datasets.keys())
