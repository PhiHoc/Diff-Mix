import abc
import math
import os
import random
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec

class SyntheticDataset(Dataset):
    def __init__(
        self,
        synthetic_dir: Union[str, List[str]] = None,
        image_size: int = 512,
        crop_size: int = 448,
        class2label: dict = None,
    ) -> None:
        super().__init__()

        self.synthetic_dir = synthetic_dir
        self.use_diffmix = False

        self.parse_synthetic_data(synthetic_dir)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.class2label = (
            {name: i for i, name in enumerate(self.class_names)}
            if class2label is None
            else class2label
        )
        self.num_classes = len(self.class2label)

    def parse_synthetic_data(self, synthetic_dir) -> None:
        if isinstance(synthetic_dir, str):
            synthetic_dir = [synthetic_dir]

        all_data = []
        class_names = set()

        for _dir in synthetic_dir:
            meta_path = os.path.join(_dir, "meta.csv")
            data_dir = os.path.join(_dir, "data")

            if os.path.exists(meta_path):
                self.use_diffmix = True
                meta_df = pd.read_csv(meta_path)
                meta_df["Path"] = meta_df["Path"].apply(lambda x: os.path.join(_dir, "data", x))
                all_data.append(meta_df)
            elif os.path.exists(data_dir):
                for class_name in os.listdir(data_dir):
                    class_path = os.path.join(data_dir, class_name)
                    if not os.path.isdir(class_path):
                        continue
                    class_names.add(class_name.replace("_", " "))
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        all_data.append({"Path": img_path, "Target Class": class_name.replace("_", " ")})
            else:
                raise ValueError(f"Synthetic dir {_dir} missing both meta.csv and data folder.")

        if self.use_diffmix:
            self.meta_df = pd.concat(all_data).reset_index(drop=True)
            self.class_names = sorted(self.meta_df["First Directory"].unique())
        else:
            self.meta_df = pd.DataFrame(all_data)
            self.class_names = sorted(list(class_names))

        print(f"Synthetic samples: {len(self.meta_df)} | DiffMix-style: {self.use_diffmix}")

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> dict:
        df_data = self.meta_df.iloc[idx]
        path = df_data["Path"]

        if self.use_diffmix:
            src_label = self.class2label[df_data["First Directory"]]
            tar_label = self.class2label[df_data["Second Directory"]]
            image = Image.open(path).convert("RGB")
            return {
                "pixel_values": self.transform(image),
                "src_label": src_label,
                "tar_label": tar_label,
            }
        else:
            label = self.class2label[df_data["Target Class"]]
            image = Image.open(path).convert("RGB")
            return {
                "pixel_values": self.transform(image),
                "label": label,
            }

class HugFewShotDataset(Dataset):
    num_classes: int = None
    class_names: int = None
    class2label: dict = None
    label2class: dict = None

    def __init__(
        self,
        split: str = "train",
        examples_per_class: int = None,
        synthetic_probability: float = 0.5,
        return_onehot: bool = False,
        soft_scaler: float = 1,
        synthetic_dir: Union[str, List[str]] = None,
        image_size: int = 512,
        crop_size: int = 448,
        gamma: int = 1,
        num_syn_seeds: int = 99999,
        clip_filtered_syn: bool = False,
        target_class_num: int = None,
        **kwargs,
    ):
        self.examples_per_class = examples_per_class
        self.num_syn_seeds = num_syn_seeds
        self.synthetic_dir = synthetic_dir
        self.clip_filtered_syn = clip_filtered_syn
        self.return_onehot = return_onehot

        if self.synthetic_dir is not None:
            self.synthetic_probability = synthetic_probability
            self.soft_scaler = soft_scaler
            self.gamma = gamma
            self.target_class_num = target_class_num
            self.parse_syn_data_pd(synthetic_dir)

        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(crop_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.transform = {"train": train_transform, "val": test_transform}[split]

    def set_transform(self, transform) -> None:
        self.transform = transform

    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:
        return NotImplemented

    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:
        return NotImplemented

    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:
        return NotImplemented

    def parse_syn_data_pd(self, synthetic_dir, filter=True) -> None:
        if isinstance(synthetic_dir, str):
            synthetic_dir = [synthetic_dir]
        meta_df_list = []
        for _dir in synthetic_dir:
            df_basename = "meta.csv" if not self.clip_filtered_syn else "remained_meta.csv"
            meta_path = os.path.join(_dir, df_basename)
            meta_df = self.filter_df(pd.read_csv(meta_path))
            meta_df["Path"] = meta_df["Path"].apply(lambda x: os.path.join(_dir, "data", x))
            meta_df_list.append(meta_df)
        self.meta_df = pd.concat(meta_df_list).reset_index(drop=True)
        self.syn_nums = len(self.meta_df)
        print(f"Syn numbers: {self.syn_nums}")
