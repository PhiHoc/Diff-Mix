import os
from datasets import load_dataset
from dataset.base import HugFewShotDataset

HUG_LOCAL_IMAGE_TRAIN_DIR = "/content/drive/MyDrive/Rare Animal/Bear/diffmix"

# Define class names (you can customize this as needed)
CLASS_NAMES = [
    "Asiatic Black Bear",
    "Kodiak Bear",
    "Andean Bear",
    "Malayan Sun Bear",
    "Sri Lankan Sloth Bear",
    "Kamchatka Brown Bear",
    "Grizzly Bear",
    "Ussuri Brown Bear",
    "Syrian Brown Bear",
    "Himalayan Brown Bear"
]

class BearHugDataset(HugFewShotDataset):
    def __init__(self, examples_per_class=-1, synthetic_probability=0.0, synthetic_dir=None, target_class_num=None):
        # Load dataset from local directory
        self.dataset = load_dataset("imagefolder", data_dir=HUG_LOCAL_IMAGE_TRAIN_DIR)["train"]
        self.num_classes = len(CLASS_NAMES)
        self.class_names = CLASS_NAMES
        self.class2label = {name: i for i, name in enumerate(CLASS_NAMES)}
        self.label2class = {i: name for i, name in enumerate(CLASS_NAMES)}

        # Build label_to_indices mapping
        self.label_to_indices = {i: [] for i in range(self.num_classes)}
        for idx, sample in enumerate(self.dataset):
            label = sample["label"]  # Integer label from directory name
            self.label_to_indices[label].append(idx)

        # Apply few-shot sampling if needed
        if examples_per_class > 0:
            self._apply_few_shot_sampling(examples_per_class)

        # Load synthetic data if provided
        if synthetic_dir is not None:
            self.parse_syn_data_pd(synthetic_dir, target_class_num)

        # Initialize base class
        super().__init__(examples_per_class, synthetic_probability, synthetic_dir, target_class_num)

    def get_image_by_idx(self, idx):
        return self.dataset[idx]["image"]

    def get_label_by_idx(self, idx):
        return self.dataset[idx]["label"]

    def get_metadata_by_idx(self, idx):
        label = self.dataset[idx]["label"]
        return {
            "class_name": self.label2class[label],
            "label": label
        }

    def __len__(self):
        return len(self.dataset)
