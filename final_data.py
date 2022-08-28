import torch

from paired import MatchedCrops
from crop import *
from cfg import *
from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms as tf



if __name__ == '__main__':


    dataset_fake = ImageDataset(file_list_fake_label,for_label=True)
    dataset_real = ImageDataset(file_list_real)
    data = MatchedCrops(dataset_fake, dataset_real, matched_crop_path, weight_path)

    loader = DataLoader(data,batch_size=2,shuffle=False)
    for i in loader:
        k = i[0]
        break
    input_label = torch.FloatTensor(2, 35, 256, 256).zero_()
    input_semantics = input_label.scatter_(1, k, 1.0)
    print(input_semantics.shape)

