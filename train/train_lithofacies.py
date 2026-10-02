"""Train InfoUNet for lithofacies prediction on synthetic 2D lines cut from the 3D volume.

Reads config/config_lithofacies.yaml (edit `main(...)` below to use another file), trains with
DistributedDataParallel on CF.GPUS, and saves the full training state every CF.TRAIN.CHECK_EPOCH
epochs to checkpoint/<CF.TAG>_<epoch:03d> (network, optimizer, lr_scheduler, train_loss and
valid_loss histories). Set CF.DATASET.TDT to train the model for a given 2D trace spacing.
Run from the repository root: python -m train.train_lithofacies
"""
import time
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yacs.config
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from module.dataset import LithofaciesDataset
from network.coordi_network import get_gen_model, DiceLoss
from utils.project import get_project_root, ValueTracker
from utils.parallel import setup, cleanup, run_target
from utils.pytorch import init_weights


def main(config):
    config_file = os.path.join(get_project_root(), "config", config)
    with open(config_file, "rt") as f_read:
        CF = yacs.config.load_cfg(f_read)

    if CF.DEVICE == "cuda":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_id) for gpu_id in CF.GPUS])

    world_size = torch.cuda.device_count() if CF.DEVICE == "cuda" else 1
    run_target(train, world_size, CF)


def train(rank, world_size, CF):
    setup(rank, world_size, 19982)

    try:
        _train(rank, world_size, CF)
    finally:
        # Always tear down the NCCL process group, even if training raised
        # (e.g. CUDA OOM) partway through -- otherwise it leaks.
        cleanup()


def _train(rank, world_size, CF):
    tag = CF.TAG
    if rank == 0:
        print("Tag:", tag)
    cudnn.benchmark = CF.CUDNN_BENCHMARK
    is_parallel = False
    if CF.DEVICE == "cpu":
        device = torch.device("cpu")
    elif CF.DEVICE == "cuda":
        if len(CF.GPUS) == 1:
            device = torch.device("cuda:" + str(CF.GPUS[0]))
        elif len(CF.GPUS) > 1:
            is_parallel = True
            device = torch.device("cuda")
            torch.cuda.set_device(rank)
        else:
            raise ValueError(f"CF.GPUS must be non-empty when CF.DEVICE == 'cuda', got: {CF.GPUS}")
    else:
        raise ValueError(f"Unsupported CF.DEVICE: {CF.DEVICE!r} (expected 'cpu' or 'cuda')")

    if rank == 0:
        print("Number of GPU:", world_size)

    train_dataset = LithofaciesDataset(
        data_pool=CF.DATASET.DATA_POOL,
        volume_tag=CF.DATASET.VOLUME_TAG,
        facies_tag=CF.DATASET.FACIES_TAG,
        target_idx=CF.DATASET.TRAIN_IDX,
        vdt=CF.DATASET.VDT,
        tdt=CF.DATASET.TDT,
        crop_size=CF.DATASET.TRAIN_CROP_SIZE,
        total_length=CF.DATASET.TRAIN_NUM_DATA,
        is_coordi=False,
        is_inst_phase=True,
        is_inst_freq=True,
        is_flip=CF.DATASET.FLIP,
        is_scale=CF.DATASET.SCALE,
        noise=CF.DATASET.NOISE,
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=CF.DATASET.TRAIN_BATCH_SIZE // world_size,
        num_workers=CF.DATASET.NUM_WORKER,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True
    )
    n_batch = len(train_loader)
    if rank == 0:
        print("Length:", n_batch)

    valid_dataset = LithofaciesDataset(
        data_pool=CF.DATASET.DATA_POOL,
        volume_tag=CF.DATASET.VOLUME_TAG,
        facies_tag=CF.DATASET.FACIES_TAG,
        target_idx=CF.DATASET.VALID_IDX,
        vdt=CF.DATASET.VDT,
        tdt=CF.DATASET.TDT,
        crop_size=CF.DATASET.VALID_CROP_SIZE,
        total_length=CF.DATASET.VALID_NUM_DATA,
        is_coordi=False,
        is_inst_phase=True,
        is_inst_freq=True,
        is_flip=False,
        is_scale=False,
        noise=0.0,
    )
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=CF.DATASET.VALID_BATCH_SIZE // world_size,
        num_workers=CF.DATASET.NUM_WORKER,
        pin_memory=True,
        sampler=valid_sampler,
        persistent_workers=True
    )

    network = get_gen_model(CF, additional_channel=0).to(rank)
    if is_parallel:
        network = DDP(network, device_ids=[rank], find_unused_parameters=True)

    # Loss = class-weighted cross-entropy + 0.1 * Dice. Classes: 0 unlabeled (ignored),
    # 1 unclassified region outside the interpreted interval (weight 0 and excluded from
    # Dice, so it never contributes to the loss), 2 basement, 3 igneous (both up-weighted
    # x2), 4 shale, 5 sand.
    class_weight = torch.tensor([0.0, 0.0, 2.0, 2.0, 1.0, 1.0], dtype=torch.float32)
    train_criterion = nn.CrossEntropyLoss(ignore_index=0, weight=class_weight, label_smoothing=0.1).to(rank)
    valid_criterion = nn.CrossEntropyLoss(ignore_index=0, weight=class_weight, label_smoothing=0.0).to(rank)
    dice_criterion = DiceLoss(ignore_index=0, ignore_classes=[1]).to(rank)
    lambda_dice = 0.1
    optimizer = optim.AdamW(network.parameters(), lr=CF.TRAIN.LR, betas=(CF.TRAIN.BETA1, CF.TRAIN.BETA2), weight_decay=0.01)
    # Cosine decay over the full run down to LR / 100.
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CF.TRAIN.END_EPOCH, eta_min=CF.TRAIN.LR / 100)

    train_losses = []
    valid_losses = []
    # Resume from checkpoint/<PRETRAIN.TAG>_<LOAD_EPOCH>, including the loss histories.
    if CF.PRETRAIN.LOAD:
        load_tag = CF.PRETRAIN.TAG
        state = torch.load(os.path.join(get_project_root(), "checkpoint", load_tag + "_" + str(CF.PRETRAIN.LOAD_EPOCH).zfill(3)), map_location=lambda storage, loc: storage, weights_only=True)

        if is_parallel:
            network.module.load_state_dict(state["network"])
        else:
            network.load_state_dict(state["network"])

        if CF.PRETRAIN.LOAD_OPTIMIZER:
            optimizer.load_state_dict(state["optimizer"])
            lr_scheduler.load_state_dict(state["lr_scheduler"])

        train_losses = state["train_loss"]
        valid_losses = state["valid_loss"]
    else:
        network.apply(init_weights)

    ema_coeff = 0.99
    avg_train_loss = ValueTracker(ema_coeff)
    for i_epoch in range(CF.TRAIN.BEGIN_EPOCH, CF.TRAIN.END_EPOCH):
        start_time = time.time()

        # Training
        # DistributedSampler shuffles using seed + epoch; without this, every
        # epoch would use the exact same shuffle order.
        train_sampler.set_epoch(i_epoch)
        avg_train_loss.initialize()
        network.train()
        for i_batch, (vt, ft, info) in enumerate(train_loader):
            vt = vt.to(rank, non_blocking=True)
            ft = ft.to(rank, non_blocking=True)
            info = info.to(rank, non_blocking=True)

            fo = network(vt, info)

            train_loss = train_criterion(fo, ft) + lambda_dice * dice_criterion(fo, ft)

            avg_train_loss.feed(train_loss.detach().item())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if rank == 0 and ((i_batch + 1) % 10 == 0 or (i_batch + 1) == n_batch):
                print("epoch: {:4}, batch: {:4}, train_loss: {:8.3e}".format(
                    i_epoch + 1,
                    i_batch + 1,
                    avg_train_loss.val(),
                ))

        # Validation
        valid_sampler.set_epoch(i_epoch)
        valid_loss_sum = 0
        network.eval()
        with torch.no_grad():
            count = 0
            for i_batch, (vt, ft, info) in enumerate(valid_loader):
                vt = vt.to(rank, non_blocking=True)
                ft = ft.to(rank, non_blocking=True)
                info = info.to(rank, non_blocking=True)

                fo = network(vt, info)

                valid_loss = valid_criterion(fo, ft) + lambda_dice * dice_criterion(fo, ft)
                torch.distributed.all_reduce(valid_loss, op=torch.distributed.ReduceOp.SUM)
                valid_loss_sum += valid_loss.detach().item()
                count += world_size
            valid_loss = valid_loss_sum / count

            if rank == 0:
                print("epoch: {:4}, valid_loss: {:8.3e}".format(
                    i_epoch + 1,
                    valid_loss,
                ))

        lr_scheduler.step()
        train_losses.append(avg_train_loss.val())
        valid_losses.append(valid_loss)

        if rank == 0:
            if (i_epoch + 1) % CF.TRAIN.CHECK_EPOCH == 0:
                state = {
                    "network": network.module.state_dict() if is_parallel else network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_loss": train_losses,
                    "valid_loss": valid_losses,
                }

                print("epoch: {:4}, save training state".format(i_epoch + 1))
                torch.save(state, os.path.join(get_project_root(), "checkpoint", tag + "_" + str(i_epoch + 1).zfill(3)))

            print("epoch: {:4}, execution time: {:6.2f}".format(
                i_epoch + 1,
                time.time() - start_time,
            ))


if __name__ == "__main__":
    main("config_lithofacies.yaml")
