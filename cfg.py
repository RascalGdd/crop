from pathlib import Path
import os

labels = []
path_lab = Path(r"/data/public/cityscapes/gtfine")
for mode in list(path_lab.iterdir()):
    if mode.stem == "test":
        continue
    path_lab_2 = Path(path_lab / mode)
    for city_folder in sorted(list(path_lab_2.iterdir())):
        cur_folder = Path(path_lab_2 / city_folder)
        for item in sorted(list(cur_folder.iterdir())):
            if "labelIds" in item.stem:
                labels.append(Path(path_lab_2 / city_folder / item))

file_list_real = labels
print(len(file_list_real))
path_folder_fake = Path("/data/public/gta/labels")
# path_folder_fake_label = Path("/data/public/gta/labels")
# path_folder_real = Path("/data/public/cityscapes/leftImg8bit/train")
# city_list = list(path_folder_real.iterdir())

file_list_fake = list(path_folder_fake.iterdir())
# file_list_fake_label = list(path_folder_fake_label.iterdir())
# file_list_real = []

# for i in city_list:
#     file_list_real += list((path_folder_real/i).iterdir())
out_dir = Path("/no_backups/s1422/cropdata_kvd")
fake_path = Path(out_dir/"crop_fake.csv")
real_path = Path(out_dir/"crop_real.csv")
fake_feature_path = Path(out_dir/"crop_fake.npz")
real_feature_path = Path(out_dir/"crop_real.npz")


match_path = Path(out_dir/"match.npz")
weight_path = Path(out_dir/"weight.npz")
matched_crop_path = Path(out_dir/"matched_crop.csv")
matched_crop_path_kvd = Path(out_dir/"matched_crop_kvd.csv")
