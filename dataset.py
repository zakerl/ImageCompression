import multiprocessing as mp
import os
import random
import sys
import time
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from image_processing import make_image_training_pair


class ImageFolderDataset(Dataset):
    def __init__(self, data_file: str, scaling_factor: int) -> None:
        self.scaling_factor = scaling_factor
        self.all_image_chunks = np.load(data_file)

    def __len__(self) -> int:
        return self.all_image_chunks.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk = self.all_image_chunks[index]
        Yx, Ux, Vx, target = make_image_training_pair(chunk, self.scaling_factor)
        return (
            torch.Tensor(Yx).float(),
            torch.Tensor(Ux).float(),
            torch.Tensor(Vx).float(),
            torch.Tensor(target).float().permute(2, 0, 1),
        )


def _load_data(root: str, target_size: int) -> np.ndarray:
    files = os.listdir(root)
    fn_args = [(os.path.join(root, file), target_size) for file in files]
    with mp.Pool() as pool:
        all_image_chunks = list(
            tqdm(pool.imap(_split_image_into_chunks, fn_args), total=len(fn_args))
        )
    return np.concatenate(all_image_chunks, axis=0)


def _split_image_into_chunks(image: Tuple[str, int]) -> np.ndarray:
    image_path, target_size = image

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    n_chunks_h = np.ceil(h / target_size).astype(int)
    n_chunks_w = np.ceil(w / target_size).astype(int)
    n_chunks = n_chunks_h * n_chunks_w

    chunks = np.zeros((n_chunks, target_size, target_size, 3), dtype=np.uint8)

    idx = 0
    for i in range(n_chunks_h):
        for j in range(n_chunks_w):
            chunk = img[
                i * target_size : (i + 1) * target_size,
                j * target_size : (j + 1) * target_size,
            ]
            h_c, w_c, _ = chunk.shape
            chunks[idx, :h_c, :w_c] = chunk
            idx += 1

    return chunks


def main(root: str, target_size: int, data_file: str) -> None:
    all_image_chunks = _load_data(root, target_size)
    np.save(data_file, all_image_chunks)


if __name__ == "__main__":
    root = sys.argv[1]
    target_size = int(sys.argv[2])
    data_file = sys.argv[3]
    main(root, target_size, data_file)
