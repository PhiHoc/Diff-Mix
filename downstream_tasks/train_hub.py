import argparse
import math
import os
import random
import shutil
import sys
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL.Image import Image
from torchvision.models import ViT_B_16_Weights, resnet18, resnet50, vit_b_16
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.base import SyntheticDataset

# sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from dataset import DATASET_NAME_MAPPING
from downstream_tasks.losses import LabelSmoothingLoss
from downstream_tasks.mixup import CutMix, mixup_data
from utils.network import freeze_model
import json
import pytorch_warmup
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset

#######################
##### 1 - Setting #####
#######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def save_metrics_to_json(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f)

def save_checkpoint(model, optimizer, epoch, metrics, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, path)


##### args setting
def formate_note(args):
    args.use_warmup = True
    note = f"{args.note}"
    if args.syndata_dir is not None:
        note = note + f"_{os.path.basename(args.syndata_dir[0])}"
    if args.use_cutmix:
        note = note + "_cutmix"
    if args.use_mixup:
        note = note + "_mixup"
    return note


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="cub", help="dataset name")
parser.add_argument("--syndata_dir", type=str, nargs="+", help="key for indexing synthetic data")
parser.add_argument("--syndata_p", default=0.1, type=float, help="synthetic probability")
parser.add_argument("-m", "--model", default="resnet50",
                    choices=["resnet18","resnet18pretrain","resnet50", "vit_b_16"], help="model name")
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch_size")
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--use_cutmix", default=False, action="store_true")
parser.add_argument("--use_mixup", default=False, action="store_true")
parser.add_argument("--criterion", default="ls", type=str)
parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU id")
parser.add_argument("-w", "--num_workers", default=4, type=int, help="num_workers of dataloader (set <=4 for Colab)")
parser.add_argument("-s", "--seed", default=2020, type=int, help="random seed")
parser.add_argument("-n", "--note", default="", help="exp note, append after exp folder")
parser.add_argument("-p", "--group_note", default="debug")
parser.add_argument("-a", "--amp", default=0, type=int, help="0: w/o amp, 1: nvidia apex.amp, 2: torch.cuda.amp")
parser.add_argument("-rs", "--resize", default=512, type=int)
parser.add_argument("--res_mode", default="224", type=str)
parser.add_argument("-cs", "--crop_size", type=int, default=448)
parser.add_argument("--examples_per_class", type=int, default=-1)
parser.add_argument("--gamma", type=float, default=1.0, help="label smoothing factor for synthetic data")
parser.add_argument("-mp", "--mixup_probability", type=float, default=0.5)
parser.add_argument("-ne", "--nepoch", type=int, default=448)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("-fs", "--finetune_strategy", type=str, default=None)
parser.add_argument("--train_data_dir", type=str, default=None, help="Path to the original training data folder.")
parser.add_argument("--test_data_dir", type=str, default=None, help="Path to the original testing data folder.")
parser.add_argument("--output_root", type=str, default="outputs/result", help="The root directory for all experiment results.")
parser.add_argument("--soft_scaler", type=float, default=1.0, help="Scaling factor for soft labels.")

parser.add_argument(
    "--data_mode",
    type=str,
    default="probabilistic",
    choices=["probabilistic", "concat"],
    help="Strategy: 'probabilistic' using syndata_p, 'concat' using combine datasets."
)

args = parser.parse_args()
run_name = f"{args.dataset}{formate_note(args)}"
exp_dir = os.path.join(args.output_root, args.group_note, run_name)

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
print(f"Thí nghiệm sẽ được lưu tại: {exp_dir}")

##### exp setting
if args.optimizer == "sgd":
    base_lr = 0.02
elif args.optimizer == "adamw":
    base_lr = 1e-3
else:
    raise ValueError("optimizer not supported")

if args.res_mode == "28":
    args.resize = 32
    args.crop_size = 28
    args.batch_size = 2048
elif args.res_mode == "224":
    args.resize = 256
    args.crop_size = 224
    if args.model == "resnet50":
        args.batch_size = 256
    elif args.model == "vit_b_16":
        args.batch_size = 128
    elif args.model == "resnet18" or args.model == "resnet18pretrain":
        args.batch_size = 256
    else:
        raise ValueError("model not supported")
elif args.res_mode == "384":
    args.resize = 440
    args.crop_size = 384
    if args.model == "resnet50":
        args.batch_size = 128
    elif args.model == "vit_b_16":
        args.batch_size = 32
    elif args.model == "resnet18" or args.model == "resnet18pretrain":
        args.batch_size = 128
    else:
        raise ValueError("model not supported")
elif args.res_mode == "448":
    args.resize = 512
    args.crop_size = 448
    if args.model == "resnet50":
        args.batch_size = 64
    elif args.model == "vit_b_16":
        args.batch_size = 32
    elif args.model == "resnet18" or args.model == "resnet18pretrain":
        args.batch_size = 64
    else:
        raise ValueError("model not supported")
else:
    raise ValueError("res_mode not supported")

# Override num_workers for Colab environment to avoid worker errors
args.num_workers = min(args.num_workers, 4)

use_amp = int(args.amp)

lr_begin = args.lr
seed = int(args.seed)
datasets_name = args.dataset
num_workers = args.num_workers

##### CUDA device setting
# Use device from torch.device
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##### Random seed setting
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# tôi đang muốn sample 1 lượt 500 ảnh mỗi class --syn_dataset_mulitiplier=5, mỗi class tôi có sẵn 100 ảnh, không phải sample nhiều lần. nhưng quá trình train_hub, tôi muốn lần lượt

##### Dataloader setting
re_size = args.resize
crop_size = args.crop_size

synthetic_dir = args.syndata_dir if args.syndata_dir is not None else None

return_onehot = True
gamma = args.gamma
synthetic_probability = args.syndata_p
examples_per_class = args.examples_per_class

def to_tensor(x):
    if isinstance(x, int):
        return torch.tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise NotImplementedError


if args.data_mode == 'probabilistic':
    print("===== Chế độ tải dữ liệu: Probabilistic (Mặc định) =====")

    # Chế độ này hoạt động như code gốc

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.stack([to_tensor(example["label"]) for example in examples])
        dtype = torch.float32 if len(labels.size()) == 2 else torch.long
        labels = labels.to(dtype=dtype)
        return {"pixel_values": pixel_values, "labels": labels}


    train_set = DATASET_NAME_MAPPING[datasets_name](
        split="train",
        image_size=re_size,
        crop_size=crop_size,
        synthetic_dir=synthetic_dir,
        synthetic_probability=synthetic_probability,
        return_onehot=return_onehot,
        gamma=gamma,
        examples_per_class=examples_per_class,
        image_train_dir=args.train_data_dir,
        image_test_dir=args.test_data_dir,
    )
    nb_class = train_set.num_classes

elif args.data_mode == 'concat':
    print("===== Chế độ tải dữ liệu: Concatenate (Tái sử dụng base.py) =====")

    # 1. Tải bộ dữ liệu gốc với HARD LABEL (dùng HugFewShotDataset)
    original_train_set = DATASET_NAME_MAPPING[datasets_name](
        split="train",
        image_size=re_size,
        crop_size=crop_size,
        return_onehot=True,
        synthetic_dir=None,  # Tắt chế độ tổng hợp để chỉ lấy ảnh gốc
        examples_per_class=examples_per_class,
        image_train_dir=args.train_data_dir,
        image_test_dir=args.test_data_dir,
    )
    nb_class = original_train_set.num_classes


    # 2. Định nghĩa lớp Dataset cho dữ liệu tổng hợp bằng cách KẾ THỪA từ SyntheticDataset
    class SyntheticSoftLabelDataset(SyntheticDataset):
        # Lớp này kế thừa toàn bộ logic đọc file và __len__ từ SyntheticDataset

        def __init__(self, *args, **kwargs):
            # Lưu lại các tham số cần cho soft label
            self.gamma = kwargs.pop('gamma', 1.0)
            self.soft_scaler = kwargs.pop('soft_scaler', 1.0)
            # Gọi hàm khởi tạo của lớp cha (SyntheticDataset)
            super().__init__(*args, **kwargs)

        def __getitem__(self, idx: int) -> dict:
            # Lấy dữ liệu thô từ lớp cha (ảnh và nhãn integer)
            path, src_label, tar_label = self.get_syn_item_raw(idx)
            image = Image.open(path).convert("RGB")

            # Đọc strength từ dataframe
            df_data = self.meta_df.iloc[idx]
            strength = df_data["Strength"]

            # Tính toán SOFT LABEL - logic tương tự HugFewShotDataset
            soft_label = torch.zeros(self.num_classes)
            soft_label[src_label] += self.soft_scaler * (1 - math.pow(strength, self.gamma))
            soft_label[tar_label] += self.soft_scaler * math.pow(strength, self.gamma)

            return {"pixel_values": self.transform(image), "label": soft_label}


    datasets_to_concat = [original_train_set]
    if synthetic_dir:
        synthetic_soft_label_set = SyntheticSoftLabelDataset(
            synthetic_dir=synthetic_dir,
            class2label=original_train_set.class2label,
            gamma=gamma,
            soft_scaler=args.soft_scaler if hasattr(args, 'soft_scaler') else 1.0,
            image_size=re_size,
            crop_size=crop_size
        )
        datasets_to_concat.append(synthetic_soft_label_set)

    # 3. Gộp các bộ dữ liệu lại
    train_set = ConcatDataset(datasets_to_concat)
    print(f"Tổng số ảnh trong bộ train sau khi gộp: {len(train_set)}")


    # 4. Collate function (cả hai dataset giờ đều trả về tensor label)
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.stack([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

# Luôn cần định nghĩa test_set
test_set = DATASET_NAME_MAPPING[datasets_name](
    split="val", image_size=re_size, crop_size=crop_size, return_onehot=return_onehot,
    image_train_dir=args.train_data_dir,
    image_test_dir=args.test_data_dir
)

batch_size = min(args.batch_size, len(train_set))

# Logic CutMix nên được áp dụng sau khi đã có train_set cuối cùng
if args.use_cutmix:
    if args.data_mode == 'concat':
        print("Lưu ý: CutMix đang được áp dụng trên bộ dữ liệu đã gộp.")
    train_set = CutMix(
        train_set, num_class=nb_class, prob=args.mixup_probability
    )

# Tạo DataLoader từ train_set đã được xác định ở trên
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,  # Dùng collate_fn tương ứng với mode
    num_workers=num_workers,
)


# Collate_fn cho eval loader (cần one-hot nếu loss function yêu cầu)
def eval_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.stack([to_tensor(example["label"]) for example in examples])
    dtype = torch.float32 if len(labels.size()) == 2 else torch.long
    labels = labels.to(dtype=dtype)
    return {"pixel_values": pixel_values, "labels": labels}


eval_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=eval_collate_fn,
    num_workers=num_workers,
)

MODEL_DICT = {
    "resnet50": resnet50,
    "resnet18": resnet18,
    "vit_b_16": vit_b_16,
}

##### Model settings
if args.model == "resnet18":
    net = resnet18(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, nb_class)
elif args.model == "resnet18pretrain":
    net = resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, nb_class)
elif args.model == "resnet50":
    net = resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, nb_class)
elif args.model == "vit_b_16":
    net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    net.heads.head = nn.Linear(net.heads.head.in_features, nb_class)

net = net.to(device)

for param in net.parameters():
    param.requires_grad = True

if args.finetune_strategy is not None and args.model == "resnet50":
    freeze_model(net, args.finetune_strategy)

##### optimizer setting
if args.criterion == "ce":
    criterion = nn.CrossEntropyLoss()
elif args.criterion == "ls":
    criterion = LabelSmoothingLoss(classes=nb_class, smoothing=0.1)
else:
    raise NotImplementedError

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr_begin)
else:
    raise ValueError("optimizer not supported")

total_steps = args.nepoch * len(train_loader.dataset) // batch_size
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
warmup_scheduler = pytorch_warmup.LinearWarmup(optimizer, warmup_period=max(int(0.1 * total_steps), 1))

##### file/folder prepare
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

shutil.copyfile(__file__, exp_dir + "/train_hub.py")

with open(os.path.join(exp_dir, "config.yaml"), "w+", encoding="utf-8") as file:
    yaml.dump(vars(args), file)

with open(os.path.join(exp_dir, "train_log.csv"), "w+", encoding="utf-8") as file:
    file.write("Epoch, lr, Train_Loss, Train_Acc, Test_Acc\n")

##### Apex or AMP setup
if use_amp == 1:
    print("\n===== Using NVIDIA AMP =====")
    from apex import amp
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
elif use_amp == 2:
    print("\n===== Using Torch AMP =====")
    scaler = GradScaler()
else:
    scaler = None

##### Resume checkpoint
checkpoint_path = os.path.join(exp_dir, "checkpoint.pth")
metrics_path = os.path.join(exp_dir, "metrics.json")

start_epoch = 0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_accuracy = 0.0

if os.path.exists(checkpoint_path):
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    metrics = checkpoint['metrics']
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    train_accuracies = metrics['train_accuracies']
    val_accuracies = metrics['val_accuracies']
    best_accuracy = metrics['best_accuracy']

    # Move optimizer state tensors to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

##### Training loop
min_train_loss = float("inf")
max_eval_acc = 0

for epoch in range(start_epoch, args.nepoch):
    print(f"\n===== Epoch: {epoch} =====")
    net.train()
    lr_now = optimizer.param_groups[0]["lr"]
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    idx = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, ncols=80)):
        idx = batch_idx
        optimizer.zero_grad()
        inputs = batch["pixel_values"].to(device)
        targets = batch["labels"].to(device)

        if args.use_mixup and np.random.rand() < args.mixup_probability:
            inputs, targets = mixup_data(inputs, targets, alpha=1.0, num_classes=nb_class)

        if inputs.shape[0] < batch_size:
            continue

        if use_amp == 1:
            with amp.scale_loss(criterion(net(inputs), targets), optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        elif use_amp == 2:
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += targets.size(0)
        if len(targets.shape) == 2:
            targets = torch.argmax(targets, axis=1)
        train_correct += predicted.eq(targets.data).cpu().sum()
        train_loss += loss.item()

        with warmup_scheduler.dampening():
            scheduler.step()

    train_acc = 100.0 * float(train_correct) / train_total
    train_loss = train_loss / (idx + 1)
    print(f"Train | lr: {lr_now:.4f} | Loss: {train_loss:.4f} | Acc: {train_acc:.3f}% ({train_correct}/{train_total})")

    ##### Evaluation every epochs
    if epoch % 1 == 0:
        net.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for _, batch in enumerate(tqdm(eval_loader, ncols=80)):
                inputs = batch["pixel_values"].to(device)
                targets = batch["labels"].to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                eval_total += targets.size(0)
                if len(targets.shape) == 2:
                    targets = torch.argmax(targets, axis=1)
                eval_correct += predicted.eq(targets.data).cpu().sum()
        eval_acc = 100.0 * float(eval_correct) / eval_total
        print(f"Test | Acc: {eval_acc:.3f}% ({eval_correct}/{eval_total})")

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(eval_acc)
        train_accuracies.append(train_acc)
        val_accuracies.append(eval_acc)

        if eval_acc > best_accuracy:
            best_accuracy = eval_acc
            print(f"New best accuracy: {best_accuracy:.2f}%")

        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "best_accuracy": best_accuracy
        }

        save_metrics_to_json(metrics, metrics_path)
        save_checkpoint(net, optimizer, epoch, metrics, checkpoint_path)

        with open(os.path.join(exp_dir, "train_log.csv"), "a+", encoding="utf-8") as file:
            file.write(f"{epoch}, {lr_now:.4f}, {train_loss:.4f}, {train_acc:.3f}%, {eval_acc:.3f}%\n")

        # Save best model
        if eval_acc > max_eval_acc:
            max_eval_acc = eval_acc
            torch.save(net.state_dict(), os.path.join(exp_dir, "max_acc.pth"), _use_new_zipfile_serialization=False)


########################
##### 3 - Testing  #####
########################
print("\n\n===== TESTING =====")

with open(os.path.join(exp_dir, "train_log.csv"), "a") as file:
    file.write("===== TESTING =====\n")

net.load_state_dict(torch.load(join(exp_dir, "max_acc.pth"), map_location=device))
net.eval()

for data_set, testloader in zip(["train", "eval"], [train_loader, eval_loader]):
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader, ncols=80)):
            inputs = batch["pixel_values"].to(device)
            targets = batch["labels"].to(device)
            if len(targets.shape) == 2:
                targets = torch.argmax(targets, axis=1)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets.data).cpu().sum()
    test_acc = 100.0 * float(test_correct) / test_total
    print(f"Dataset {data_set}\tACC:{test_acc:.2f}%")

    with open(os.path.join(exp_dir, "train_log.csv"), "a+", encoding="utf-8") as file:
        file.write(f"Dataset {data_set}\tACC:{test_acc:.2f}\n")

    with open(os.path.join(exp_dir, f"acc_{data_set}_{test_acc:.2f}"), "a+", encoding="utf-8") as file:
        # save accuracy as file name (empty)
        pass
