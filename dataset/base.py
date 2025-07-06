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
        gamma: int = 1,
        soft_scaler: float = 1,
        num_syn_seeds: int = 999,
        image_size: int = 512,
        crop_size: int = 448,
        class2label: dict = None,
    ) -> None:
        super().__init__()

        self.synthetic_dir = synthetic_dir
        self.num_syn_seeds = num_syn_seeds
        self.gamma = gamma
        self.soft_scaler = soft_scaler
        self.class_names = None
        self.is_diffmix = False

        self.parse_syn_data_pd(synthetic_dir)

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
        self.num_classes = len(self.class2label.keys())

    def set_transform(self, transform) -> None:
        self.transform = transform

    def parse_syn_data_pd(self, synthetic_dir) -> None:
        if isinstance(synthetic_dir, str):
            synthetic_dir = [synthetic_dir]
        elif not isinstance(synthetic_dir, list):
            raise NotImplementedError("Unsupported type for synthetic_dir")

        meta_df_list = []

        for _dir in synthetic_dir:
            meta_csv_path = os.path.join(_dir, "meta.csv")
            data_dir_path = os.path.join(_dir, "data")

            if os.path.exists(meta_csv_path):
                self.is_diffmix = True
                meta_df = pd.read_csv(meta_csv_path)
                meta_df["Path"] = meta_df["Path"].apply(lambda x: os.path.join(_dir, "data", x))
                meta_df_list.append(meta_df)
            elif os.path.exists(data_dir_path):
                self.is_diffmix = False
                data = []
                class_names = []
                for class_name in os.listdir(data_dir_path):
                    class_path = os.path.join(data_dir_path, class_name)
                    if not os.path.isdir(class_path):
                        continue
                    class_names.append(class_name.replace("_", " "))
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        data.append({"Path": img_path, "Target Class": class_name.replace("_", " ")})
                meta_df = pd.DataFrame(data)
                meta_df_list.append(meta_df)
            else:
                raise ValueError(f"Synthetic directory {_dir} missing both meta.csv and data folder.")

        self.meta_df = pd.concat(meta_df_list).reset_index(drop=True)

        if self.is_diffmix:
            self.class_names = list(set(self.meta_df["First Directory"].values))
        else:
            self.class_names = sorted(self.meta_df["Target Class"].unique())

        print(f"Synthetic samples: {len(self.meta_df)} | DiffMix-style: {self.is_diffmix}")

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> dict:
        df_data = self.meta_df.iloc[idx]
        path = df_data["Path"]

        if self.is_diffmix:
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
    def __init__(
        self,
        root_dir: str,
        image_size: int = 512,
        crop_size: int = 448,
        class2label: dict = None,
    ) -> None:
        super().__init__()

        self.root_dir = root_dir

        class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_names = sorted([name.replace("_", " ") for name in class_names])
        self.class2label = (
            {name: i for i, name in enumerate(self.class_names)}
            if class2label is None
            else class2label
        )

        self.samples = []
        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.samples.append({"Path": img_path, "Target Class": class_name.replace("_", " ")})

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.num_classes = len(self.class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        path = sample["Path"]
        label = self.class2label[sample["Target Class"]]
        image = Image.open(path).convert("RGB")

        return {
            "pixel_values": self.transform(image),
            "label": label,
        }