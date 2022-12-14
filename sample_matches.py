import argparse
import csv
from pathlib import Path
import random
import os
from imageio import imwrite
import numpy as np
import torch
from torchvision.utils import make_grid
from crop import *
from cfg import *


# for each threshold,
# find a couple of samples
# load them
# make a sample picture

def load_crops(path):
    paths = []
    coords = []
    with open(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            paths.append(row['path'])
            coords.append((int(row['r0']), int(row['r1']), int(row['c0']), int(row['c1'])))
            pass
        pass
    return paths, coords


if __name__ == '__main__':

    src_img_path = file_list_fake
    dst_img_path = file_list_real
    match_path = match_path
    src_crop_path = fake_path
    dst_crop_path = real_path

    src_dataset = ImageDataset(src_img_path)
    dst_dataset = ImageDataset(dst_img_path)

    data = np.load(match_path)
    s = data['dist']  # [:,0]

    src_paths, src_coords = load_crops(src_crop_path)
    dst_paths, dst_coords = load_crops(dst_crop_path)

    dst_id = data['ind']
    thresholds = [0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 1.0, 1.2, 1.5, 2.0]
    for ti, t in enumerate(thresholds[1:]):

        print(f'Sampling dist at {t}...')
        src_id, knn = np.nonzero(np.logical_and(thresholds[ti] < s, s < t))
        crops = []
        rd = np.random.permutation(src_id.shape[0])
        for x in range(min(25, src_id.shape[0])):
            i = int(rd[x])
            print(f'\tloading sample {i}...')
            img, _ = src_dataset.get_by_path(src_paths[int(src_id[i])])
            r0, r1, c0, c1 = src_coords[int(src_id[i])]
            a = img[:, r0:r1, c0:c1].unsqueeze(0)
            img, _ = dst_dataset.get_by_path(dst_paths[int(dst_id[int(src_id[i]), int(knn[i])])])
            r0, r1, c0, c1 = dst_coords[int(dst_id[int(src_id[i]), int(knn[i])])]
            b = img[:, r0:r1, c0:c1].unsqueeze(0)
            crops.append(a)
            crops.append(b)
            pass

        if len(crops) > 0:
            grid = make_grid(torch.cat(crops, 0), nrow=2)
            imwrite(f'knn_{t}.jpg', (255.0 * grid.permute(1, 2, 0).numpy()).astype(np.uint8))
            pass
        pass
    pass