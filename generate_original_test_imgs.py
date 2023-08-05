import sys
import math
import torch
from skimage import io
from datetime import datetime

import pathlib
from load_checkpoint import load_checkpoint
from utils import *


if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

def generate_imgs_with_model(path_dir, sample_dataloader, mask_text_dataset, batch_size):
    path_dir.mkdir(parents=True, exist_ok=True)

    mask_text_dataset.apply_transforms = False
    sample_iterator = iter(sample_dataloader)
    num_total_batches = math.ceil(mask_text_dataset.__len__()*0.1/batch_size)
    i = 0
    while True:
        time_str = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        print(f"{time_str} Batch {i:>6}/{num_total_batches}", end="\r")
        try:
            images, masks, conditional_text_embedding, idx = next(sample_iterator)
        except StopIteration:
            break
        save_imgs_from_batch(images, idx, path_dir)
        i = i+1

def save_imgs_from_batch(imgs, idx, path_dir):
    for index in range(imgs.shape[0]):
        img = imgs[index]
        img = img.permute(1,2,0).cpu()
        
        img = np.array(img)
        img = np.round(img*255)
        img = img.astype(np.uint8)
        path_img = path_dir / f"{idx[index]}"
        io.imsave(path_img, img)

def generate_imgs_for_dir(path_model_dir, path_output_dir):
    path_chkpt = path_model_dir / pathlib.Path("ckpt_dict.pt")

    if not path_chkpt.exists():
        print(f"Error: Path {path_chkpt} does not exist!")
        return

    _, _, sample_dataloader, mask_text_dataset, _, args, _ = load_checkpoint(path_chkpt=path_chkpt)
    generate_imgs_with_model(path_output_dir, sample_dataloader, mask_text_dataset, args.batch_size)


if __name__ == "__main__":
    path_model_dir = Path("results/inpainting/")
    path_output_dir = Path("data/CelebAMask-HQ_test_imgs/")
    generate_imgs_for_dir(path_model_dir, path_output_dir)