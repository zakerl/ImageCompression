import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import ImageFolderDataset
from super_resolution import SuperResolution
from vdsr_model import VDSR


def main() -> None:
    torch.set_float32_matmul_precision("medium")

    train_dataset = ImageFolderDataset(
        data_file="train.npy",
        scaling_factor=2,
    )
    valid_dataset = ImageFolderDataset(
        data_file="valid.npy",
        scaling_factor=2,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=16,
        shuffle=True,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=64,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
    )

    sr = SuperResolution(
        model=VDSR(
            in_channels=3,
            out_channels=3,
            num_layers=20,
            channels=64,
        ),
        factor=2,
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir="logs",
        name="vdsr",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="deepspeed_stage_2",
        max_epochs=3,
        precision="16-mixed",
        enable_checkpointing=False,
        logger=logger,
    )
    trainer.fit(
        sr,
        train_dataloader,
        valid_dataloader,
    )

    torch.save(sr.model.state_dict(), "super_resolution.pth")


if __name__ == "__main__":
    main()
