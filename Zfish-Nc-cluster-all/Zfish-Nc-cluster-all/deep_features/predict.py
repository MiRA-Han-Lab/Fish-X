import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
import warnings
import trimesh

from tqdm import tqdm
from attrdict import AttrDict
from collections import OrderedDict
from skimage import morphology

import torch.nn as nn
import torch.nn.functional as F

from data_utils.pcloader import ModelNetDataLoader
from model.pointnetv2_encoder import PointNetV2

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model(ckpt_path, device):
    print("Loading PointNetV2...")
    model = PointNetV2().to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_weights"]

    # 去掉多卡训练时的 `module.` 前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def extract_features(cfg, model, device, out_path):
    root = "/data1/share/SCN/zhangyc/zfish_cloudvolume/pc2/"
    dataset = ModelNetDataLoader(root, npoint=3072, uniform=False, istrain=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=64)

    print("Total samples:", len(dataset))

    features, keys = [], []
    pbar = tqdm(total=len(dataset), desc="Extracting features")

    with torch.no_grad():
        for data in dataloader:
            inputs = data.to(device)
            representation = model(inputs).cpu().numpy()
            features.extend(representation)
            pbar.update(len(inputs))

    pbar.close()

    keys = [int(i.split(".")[0]) for i in dataset._list()]

    np.save(os.path.join(out_path, "keys_soma2.npy"), np.array(keys))
    np.save(os.path.join(out_path, "features_soma2.npy"), np.array(features))
    print(f"Saved features to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cfg", type=str, default="pretraining_pointcloud", help="path to config file"
    )
    parser.add_argument("-m", "--mode", type=str, default="", help="running mode")
    parser.add_argument(
        "-s", "--save", action="store_false", default=True, help="save extracted features"
    )
    args = parser.parse_args()

    cfg_file = args.cfg + ".yaml"
    print("Using config:", cfg_file)

    with open(os.path.join("./config", cfg_file), "r") as f:
        cfg = AttrDict(yaml.safe_load(f))

    device = torch.device("cuda:0")

    ckpt_path = "./trained_model/model-075000-0617.ckpt"
    model = load_model(ckpt_path, device)

    out_path = "./features_output"
    os.makedirs(out_path, exist_ok=True)

    extract_features(cfg, model, device, out_path)

    print("Done.")
