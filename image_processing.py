import math

import cv2
import numpy as np
from numba import jit


@jit
def upSample(chan, factor):
    old_h, old_w = chan.shape
    new_h = old_h * factor
    new_w = old_w * factor
    resized_image = np.zeros(shape=(new_h, new_w), dtype=np.float32)
    for i in range(new_h):
        for j in range(new_w):
            weightSum = 0
            old_y = i / factor
            old_x = j / factor
            prev_x = math.floor(old_x)
            next_x = min(old_w - 1, math.ceil(old_x))
            prev_y = math.floor(old_y)
            next_y = min(old_h - 1, math.ceil(old_y))
            if (prev_x == next_x) and (prev_y == next_y):
                TrueIntR = chan[prev_y, prev_x]
            elif prev_x == next_x:
                r1 = chan[prev_y, prev_x]
                r2 = chan[next_y, prev_x]
                TrueIntR = (next_y - old_y) / (next_y - prev_y) * r2 + (
                    old_y - prev_y
                ) / (next_y - prev_y) * r1
            elif prev_y == next_y:
                r1 = chan[prev_y, prev_x]
                r2 = chan[prev_y, next_x]
                TrueIntR = (next_x - old_x) / (next_x - prev_x) * r2 + (
                    old_x - prev_x
                ) / (next_x - prev_x) * r1
            else:
                R1 = chan[prev_y, prev_x]
                R2 = chan[prev_y, next_x]
                R3 = chan[next_y, prev_x]
                R4 = chan[next_y, next_x]
                UppInt_R = (next_x - old_x) / (next_x - prev_x) * R2 + (
                    old_x - prev_x
                ) / (next_x - prev_x) * R1
                LowInt_R = (next_x - old_x) / (next_x - prev_x) * R4 + (
                    old_x - prev_x
                ) / (next_x - prev_x) * R3
                TrueIntR = (next_y - old_y) / (next_y - prev_y) * LowInt_R + (
                    old_y - prev_y
                ) / (next_y - prev_y) * UppInt_R
            resized_image[i, j] = TrueIntR

    return resized_image


@jit
def downSample(chan, factor):
    original_width = chan.shape[1]
    original_height = chan.shape[0]
    width = original_width // factor
    height = original_height // factor
    resized_image = np.zeros(shape=(height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            currPixel = 0
            for x in range(factor):
                for y in range(factor):
                    currPixel += chan[i * factor + x, j * factor + y]
            resized_image[i, j] = currPixel / (factor**2)

    return resized_image


def rgb_to_yuv(x):
    R, G, B = np.moveaxis(x.astype(np.float32), -1, 0)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.147 * R - 0.289 * G + 0.436 * B
    V = 0.615 * R - 0.515 * G - 0.100 * B
    return np.moveaxis(np.array([Y, U, V]), 0, -1)


def yuv_to_rgb(x):
    Y, U, V = np.moveaxis(x.astype(np.float32), -1, 0)
    R = Y + 1.13983 * V
    G = Y - 0.39465 * U - 0.58060 * V
    B = Y + 2.03211 * U
    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)
    return np.moveaxis(np.array([R, G, B]), 0, -1)


def PSNR(old_image, new_image):
    MSE = np.mean((old_image - new_image) ** 2)
    return 20 * math.log10(255 / math.sqrt(MSE))


def normalize_yuv(yuv_image):
    yuv_image[..., :] += [0, 110.744, 156.21]
    yuv_image[..., :] /= [254, 221.488, 312.42]


def unnormalize_yuv(yuv_image):
    yuv_image[..., :] *= [254, 221.488, 312.42]
    yuv_image[..., :] -= [0, 110.744, 156.21]


def make_image_training_pair(image, factor):
    yuv_image = rgb_to_yuv(image)
    normalize_yuv(yuv_image)

    Y = yuv_image[..., 0]
    U = yuv_image[..., 1]
    V = yuv_image[..., 2]
    Yx = downSample(Y, factor)
    Ux = downSample(U, factor * 2)
    Vx = downSample(V, factor * 2)

    return Yx, Ux, Vx, yuv_image


def ssim_single_channel(img1, img2, C1=6.5025, C2=58.5225):
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.ravel(), img2.ravel())[0][1]

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_value = numerator / denominator

    return ssim_value


def ssim(img1, img2):
    img1_channels = cv2.split(img1)
    img2_channels = cv2.split(img2)

    ssim_values = [
        ssim_single_channel(img1_channels[i], img2_channels[i]) for i in range(3)
    ]

    avg_ssim = np.mean(ssim_values)

    return avg_ssim
