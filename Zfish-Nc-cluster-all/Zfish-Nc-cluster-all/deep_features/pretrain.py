from __future__ import absolute_import, division, print_function

import os
import sys
import time
import yaml
import torch
import logging
import argparse
import numpy as np

from collections import OrderedDict
from attrdict import AttrDict
from torch import nn
from tensorboardX import SummaryWriter

from data_utils.pcloader import Provider
from model.pointnetv2_encoder import PointNetV2
from byol_simsiam import BYOL


# ----------------------------------------------------------------------------- #
# Utils
# ----------------------------------------------------------------------------- #
def init_logging(log_path: str):
    """Initialize logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%m-%d %H:%M",
        filename=log_path,
        filemode="w"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("").addHandler(console)


def init_project(cfg):
    """Prepare folders, logging, tensorboard writer."""
    if cfg.TRAIN.if_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but `if_cuda` is set to True.")

    prefix = cfg.time
    model_name = cfg.TRAIN.model_name if cfg.TRAIN.resume else f"{prefix}_{cfg.NAME}"

    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, "valid")

    if not cfg.TRAIN.resume:
        for path in [cfg.cache_path, cfg.save_path, cfg.record_path, cfg.valid_path]:
            os.makedirs(path, exist_ok=True)

    init_logging(os.path.join(cfg.record_path, f"{prefix}.log"))
    logging.info("========== Config ==========")
    logging.info(cfg)
    logging.info("============================")

    writer = SummaryWriter(cfg.record_path)
    writer.add_text("cfg", str(cfg))
    return writer


# ----------------------------------------------------------------------------- #
# Dataset & Model
# ----------------------------------------------------------------------------- #
def load_dataset(cfg):
    logging.info("Caching datasets ...")
    t1 = time.time()

    if not hasattr(cfg, "DATASET") or not hasattr(cfg.DATASET, "root"):
        raise AttributeError("Please specify cfg.DATASET.root in your config yaml")

    train_provider = Provider(cfg, cfg.DATASET.root, n_point=cfg.DATASET.n_point)

    logging.info("Loaded dataset from %s (n_point=%d)", cfg.DATASET.root, cfg.DATASET.n_point)
    logging.info("Done (time: %.2fs)", time.time() - t1)
    return train_provider


def build_model(cfg):
    logging.info("Building model ...")
    t1 = time.time()
    device = torch.device("cuda:0")

    model = PointNetV2().to(device)

    if cfg.MODEL.pre_train:
        ckpt_path = cfg.MODEL.pretrain_path
        checkpoint = torch.load(ckpt_path, map_location=device)

        state_dict = checkpoint["model_weights"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        logging.info("Loaded pretrained weights from %s", ckpt_path)

    if torch.cuda.device_count() > 1:
        if cfg.TRAIN.batch_size % torch.cuda.device_count() != 0:
            raise ValueError(
                f"Batch size ({cfg.TRAIN.batch_size}) not divisible by number of GPUs ({torch.cuda.device_count()})"
            )
        model = nn.DataParallel(model)
        logging.info("Using %d GPUs.", torch.cuda.device_count())
    else:
        logging.info("Using a single GPU.")

    logging.info("Done (time: %.2fs)", time.time() - t1)
    return model


# ----------------------------------------------------------------------------- #
# Training
# ----------------------------------------------------------------------------- #
def resume_params(cfg, model, optimizer, resume):
    if not resume:
        return model, optimizer, 0

    model_path = os.path.join(cfg.save_path, f"model-{cfg.TRAIN.model_id:06d}.ckpt")
    logging.info("Resuming weights from %s ...", model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No checkpoint found at {model_path}")

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_weights"])
    logging.info("Resumed at iteration %d", checkpoint["current_iter"])
    return model, optimizer, checkpoint["current_iter"]


def calculate_lr(cfg, iters):
    """Polynomial decay learning rate schedule with warmup."""
    if iters < cfg.TRAIN.warmup_iters:
        return (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
            float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power
        ) + cfg.TRAIN.end_lr
    if iters < cfg.TRAIN.decay_iters:
        return (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
            1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power
        ) + cfg.TRAIN.end_lr
    return cfg.TRAIN.end_lr


def loop(cfg, train_provider, learner, optimizer, iters, writer):
    loss_file = open(os.path.join(cfg.record_path, "loss.txt"), "a")
    rcd_time, sum_time, sum_loss = [], 0, 0

    while iters <= cfg.TRAIN.total_iters:
        learner.train()
        iters += 1
        t1 = time.time()
        input1, input2 = train_provider.next()

        # Learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(cfg, iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        # Forward & backward
        optimizer.zero_grad()
        loss = learner(input1, input2)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        sum_time += time.time() - t1

        # Logging
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            avg_loss = sum_loss / max(1, (cfg.TRAIN.display_freq if iters > 1 else 1))
            logging.info(
                "step %d, loss = %.6f, lr: %.6f, et: %.2fs, rd: %.2fmin",
                iters, avg_loss, current_lr, sum_time,
                (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60,
            )
            writer.add_scalar("loss", avg_loss, iters)
            loss_file.write(f"step={iters}, loss={avg_loss:.6f}\n")
            loss_file.flush()

            rcd_time.append(sum_time)
            sum_time, sum_loss = 0, 0

        # Save checkpoint
        if iters % cfg.TRAIN.save_freq == 0:
            states = {"current_iter": iters, "model_weights": learner.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, f"model-{iters:06d}.ckpt"))
            logging.info("*************** Saved model @ step %d ***************", iters)

    loss_file.close()


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, default="pretraining_pointcloud", help="path to config file")
    parser.add_argument("-m", "--mode", type=str, default="train", help="mode: train / test")
    args = parser.parse_args()

    cfg_file = f"{args.cfg}.yaml"
    with open(os.path.join("./config", cfg_file), "r") as f:
        cfg = AttrDict(yaml.safe_load(f))

    cfg.time = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
    cfg.path = cfg_file

    if args.mode == "train":
        writer = init_project(cfg)
        train_provider = load_dataset(cfg)
        model = build_model(cfg)

        learner = BYOL(
            net=model,
            hidden_layer=-1,
            use_momentum=False,
            projection_size=512,
            projection_hidden_size=512,
        )

        optimizer = torch.optim.AdamW(
            learner.parameters(),
            lr=cfg.TRAIN.base_lr,
            betas=(0.9, 0.999),
            eps=0.01,
            weight_decay=1e-6,
            amsgrad=True,
        )

        learner, optimizer, init_iters = resume_params(cfg, learner, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, learner, optimizer, init_iters, writer)
        writer.close()

    logging.info("*** Done ***")
