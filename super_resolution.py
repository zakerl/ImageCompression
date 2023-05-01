import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import StructuralSimilarityIndexMeasure


class SuperResolution(pl.LightningModule):
    def __init__(self, model: nn.Module, factor: int):
        super().__init__()
        self.model = model
        self.factor = factor
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    @staticmethod
    def psnr(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        se = (output - target) ** 2
        se = se.mean(dim=(1, 2, 3)) + 1e-4
        return (10 * torch.log10(1 / se)).mean()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def _step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Yx, Ux, Vx, image_y = batch
        image_y_hat = self(Yx, Ux, Vx, self.factor)

        loss = F.mse_loss(image_y_hat, image_y, reduction="none").sum(dim=1).mean()

        with torch.no_grad():
            model_psnr = self.psnr(image_y_hat, image_y)
            model_ssim = self.ssim(image_y_hat, image_y)

        return loss, model_psnr, model_ssim

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, model_psnr, model_ssim = self._step(batch)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_psnr",
            model_psnr,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_ssim",
            model_ssim,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, model_psnr, model_ssim = self._step(batch)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_psnr",
            model_psnr,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_ssim",
            model_ssim,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.RAdam(
            self.parameters(),
            lr=5e-4,
        )
