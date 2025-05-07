import random
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image

from dataset.base import HugFewShotDataset
from dataset.template import IMAGENET_TEMPLATES_TINY

HUG_LOCAL_IMAGE_TRAIN_DIR = "/content/drive/MyDrive/RareAnimal/Bear/diffmix"

class BearHugDataset(HugFewShotDataset):
    super_class_name = "bear"

    def __init__(
        self,
        *args,
        split: str = "train",
        seed: int = 0,
        image_train_dir: str = HUG_LOCAL_IMAGE_TRAIN_DIR,
        examples_per_class: int = -1,
        synthetic_probability: float = 0.5,
        return_onehot: bool = False,
        soft_scaler: float = 0.9,
        synthetic_dir: str = None,
        image_size: int = 512,
        crop_size: int = 448,
        **kwargs,
    ):
        super().__init__(
            *args,
            split=split,
            examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            return_onehot=return_onehot,
            soft_scaler=soft_scaler,
            synthetic_dir=synthetic_dir,
            image_size=image_size,
            crop_size=crop_size,
            **kwargs,
        )

        dataset = load_dataset("imagefolder", data_dir=image_train_dir)["train"]
        random.seed(seed)
        np.random.seed(seed)

        # Sample few-shot if needed
        if examples_per_class is not None and examples_per_class > 0:
            all_labels = dataset["label"]
            label_to_indices = defaultdict(list)
            for i, label in enumerate(all_labels):
                label_to_indices[label].append(i)

            _all_indices = []
            for key, items in label_to_indices.items():
                try:
                    sampled_indices = random.sample(items, examples_per_class)
                except ValueError:
                    print(f"{key}: Sample larger than population or is negative, using random.choices instead")
                    sampled_indices = random.choices(items, k=examples_per_class)

                _all_indices.extend(sampled_indices)
            dataset = dataset.select(_all_indices)

        self.dataset = dataset
        self.class_names = [name.replace("/", " ") for name in dataset.features["label"].names]
        self.num_classes = len(self.class_names)
        self.class2label = self.dataset.features["label"]._str2int
        self.label2class = {v: k.replace("/", " ") for k, v in self.class2label.items()}

        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset["label"]):
            self.label_to_indices[label].append(i)

    def __len__(self):
        return len(self.dataset)

    def get_image_by_idx(self, idx: int) -> Image.Image:
        return self.dataset[idx]["image"].convert("RGB")

    def get_label_by_idx(self, idx: int) -> int:
        return self.dataset[idx]["label"]

    def get_metadata_by_idx(self, idx: int) -> dict:
        return dict(
            name=self.label2class[self.get_label_by_idx(idx)],
            super_class=self.super_class_name,
        )

    def __getitem__(self, idx: int) -> dict:
        image = self.get_image_by_idx(idx)
        label = self.get_label_by_idx(idx)
        class_name = self.label2class[label]

        # Generate a simple placeholder caption
        caption = f"An image of a {class_name}."

        return {
            "image": image,
            "label": label,
            "caption": caption,  # Add placeholder caption
        }
