import numpy as np
import torch

from paired import MatchedCrops
from crop import *
from cfg import *
from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms as tf
from torchvision.utils import save_image


if __name__ == '__main__':


    dataset_fake = ImageDataset(file_list_fake, for_label=True)
    dataset_fake2 = ImageDataset(file_list_fake, for_label=True)
    dataset_real = ImageDataset(file_list_real, for_label=True)
    data = MatchedCrops(dataset_fake, dataset_real,dataset_fake2, matched_crop_path_kvd, weight_path)

    loader = DataLoader(data,batch_size=2,shuffle=True)
    for idx, i in enumerate(loader):
        k = i[0][0][0]
        j = i[1][0][0]
        m = i[2]
        # k = k/255
        # j = j/255
        # m = m/255
        # k, j, m = k.double(), j.double(), m.double()
        # save_image(k, out_dir / (str(idx)+"label"+".jpg"))
        # save_image(j, out_dir / (str(idx) + "real" + ".jpg"))
        # save_image(m, out_dir / (str(idx) + "fake" + ".jpg"))
        k = np.array(k)
        j = np.array(j)
        print(np.sum(k == j) / 256 ** 2)

    # input_label = torch.FloatTensor(2, 35, 256, 256).zero_()
    # input_semantics = input_label.scatter_(1, k, 1.0)
    # print(input_semantics.shape)

