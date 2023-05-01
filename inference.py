import sys

import cv2
import numpy as np
import torch

from image_processing import (
    PSNR,
    downSample,
    normalize_yuv,
    rgb_to_yuv,
    ssim,
    unnormalize_yuv,
    upSample,
    yuv_to_rgb,
)
from vdsr_model import VDSR


@torch.no_grad()
def main() -> None:
    torch.set_float32_matmul_precision("medium")

    filename = sys.argv[1]

    model = VDSR(
        in_channels=3,
        out_channels=3,
        num_layers=20,
        channels=64,
    )
    model.load_state_dict(torch.load(
        "super_resolution.pth", map_location="cpu"))
    model = model.cuda().eval()

    image = cv2.imread(filename)
    orig_image = image.copy()
    orig_size = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.copyMakeBorder(
        image,
        0,
        (4 - image.shape[0] % 4) % 4,
        0,
        (4 - image.shape[1] % 4) % 4,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    image = rgb_to_yuv(image)
    normalize_yuv(image)
    Y, U, V = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    Y = downSample(Y, 2)
    U = downSample(U, 4)
    V = downSample(V, 4)
    cv2.imwrite("Y.png", (Y*255).astype(np.uint8))
    cv2.imwrite("U.png", (U*255).astype(np.uint8))
    cv2.imwrite("V.png", (V*255).astype(np.uint8))
    Yx = torch.Tensor(Y[None]).cuda()
    Ux = torch.Tensor(U[None]).cuda()
    Vx = torch.Tensor(V[None]).cuda()
    model_upsampled = model(Yx, Ux, Vx, 2)[0].cpu().permute((1, 2, 0)).numpy()

    naive_upsampled_y = upSample(Y, 2)
    naive_upsampled_u = upSample(U, 4)
    naive_upsampled_v = upSample(V, 4)
    naive_upsampled = np.stack(
        [naive_upsampled_y, naive_upsampled_u, naive_upsampled_v], axis=2
    )

    unnormalize_yuv(model_upsampled)
    rgb_model_upsampled = yuv_to_rgb(model_upsampled)

    unnormalize_yuv(naive_upsampled)
    rgb_naive_upsampled = yuv_to_rgb(naive_upsampled)

    model_upsampled = cv2.cvtColor(rgb_model_upsampled, cv2.COLOR_RGB2BGR)
    naive_upsampled = cv2.cvtColor(rgb_naive_upsampled, cv2.COLOR_RGB2BGR)

    model_upsampled = model_upsampled[: orig_size[0], : orig_size[1]]
    naive_upsampled = naive_upsampled[: orig_size[0], : orig_size[1]]

    cv2.imwrite("model_upsampled.png", model_upsampled)
    cv2.imwrite("naive_upsampled.png", naive_upsampled)

    model_v_orig_psnr = PSNR(orig_image, model_upsampled)
    naive_v_orig_psnr = PSNR(orig_image, naive_upsampled)

    model_v_orig_ssim = ssim(orig_image, model_upsampled)
    naive_v_orig_ssim = ssim(orig_image, naive_upsampled)

    print(f"Model PSNR: {model_v_orig_psnr}")
    print(f"Naive PSNR: {naive_v_orig_psnr}")

    print(f"Model SSIM: {model_v_orig_ssim}")
    print(f"Naive SSIM: {naive_v_orig_ssim}")


if __name__ == "__main__":
    main()
