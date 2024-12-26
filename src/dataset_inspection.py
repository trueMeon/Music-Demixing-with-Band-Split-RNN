import argparse

from pathlib import Path
from typing import List, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from typing import Tuple
import shutil

import torch
import torch.nn as nn
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from tqdm import tqdm

from data import SourceSeparationDataset, collate_fn, PreloadSourceSeparationDataset
from model import BandSplitRNN, PLModel

import logging
import traceback

import utils

from pytorch_lightning.loggers import WandbLogger

log = logging.getLogger(__name__)


def initialize_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Initializes train and validation dataloaders from configuration file.
    """
    train_dataset = SourceSeparationDataset(
        **cfg.train_dataset,
    )
    train_loader = DataLoader(
        train_dataset,
        **cfg.train_loader,
        collate_fn=collate_fn
    )
    if hasattr(cfg, 'val_dataset'):
        val_dataset = SourceSeparationDataset(
            **cfg.val_dataset,
        )
        val_loader = DataLoader(
            val_dataset,
            **cfg.val_loader,
            collate_fn=collate_fn
        )
    else:
        val_loader = None
    return (
        train_loader,
        val_loader
    )

def initialize_preloaded_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    train_dataset = PreloadSourceSeparationDataset(
        **cfg.train_dataset,
    )
    train_loader = DataLoader(
        train_dataset,
        **cfg.train_loader,
        collate_fn=collate_fn
    )

    val_loader = None

    if hasattr(cfg, 'val_dataset'):
        val_dataset = PreloadSourceSeparationDataset(
            **cfg.val_dataset,
        )
        val_loader = DataLoader(
            val_dataset,
            **cfg.val_loader,
            collate_fn=collate_fn
        )

    return (
        train_loader,
        val_loader
    )

@hydra.main(version_base=None, config_path="conf", config_name="config")
def inspect(
        # device: str,
        cfg: DictConfig
) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    log.info(f"Float32 matmul precision: {cfg.torch.matmul_precision}")
    torch.set_float32_matmul_precision(cfg.torch.matmul_precision)
    
    pl.seed_everything(42, workers=True)

    # log.info(OmegaConf.to_yaml(cfg))

    log.info("Initializing loaders")
    train_loader, val_loader = initialize_preloaded_loaders(cfg)
    # train_loader, val_loader = initialize_loaders(cfg)
    log.info(f"Dataset length: {len(train_loader)}")
    log.info(f"Batch size: {train_loader.batch_size}")
    output_dir_path = Path("files/output_test")

    if output_dir_path.exists():
        shutil.rmtree(output_dir_path)
    
    for batch_num, batch in tqdm(enumerate(train_loader)):
        for element_num in range(batch.shape[0]):
            mix_seg = batch[element_num, 0, :, :]
            tgt_seg = batch[element_num, 1, :, :]
            
            output_sample_dir_path = output_dir_path / f"{batch_num}_{element_num}"
            output_sample_dir_path.mkdir(parents=True, exist_ok=True)

            mix_path = output_sample_dir_path / "mix.wav"
            tgt_path = output_sample_dir_path / "tgt.wav"

            torchaudio.save(mix_path, src=mix_seg, sample_rate=cfg.train_dataset.sr, channels_first=True)
            torchaudio.save(tgt_path, src=tgt_seg, sample_rate=cfg.train_dataset.sr, channels_first=True)

if __name__ == "__main__":
    inspect()
