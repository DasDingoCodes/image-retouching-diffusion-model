import sys
import math
import torch
from skimage import io
from datetime import datetime
import torchvision.transforms as T

import pathlib
from load_checkpoint import load_checkpoint
from utils import *


if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

def generate_imgs_with_model(path_parent_dir, model, sample_dataloader, mask_text_dataset, diffusion, args, guidance_scale = 0):

    path_output_dir = path_parent_dir / f"test_imgs/guidance_scale_{guidance_scale}"
    path_output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_file_count = len([x for x in path_output_dir.iterdir()])
    if output_dir_file_count -1 <= mask_text_dataset.__len__()*0.1 and output_dir_file_count+1 >= mask_text_dataset.__len__()*0.1:
        print(f"There already are {output_dir_file_count} files in dir {path_output_dir}.")
        print("Seems like the images were already generated for that guidance scale.")
        print("Abort.")
        return

    mask_text_dataset.apply_transforms = False

    if torch.cuda.device_count() > 1:
        device = args.device
    else:
        device = "cuda"

    sample_iterator = iter(sample_dataloader)

    num_total_batches = math.ceil(mask_text_dataset.__len__()*0.1/args.batch_size)
    i = 0
    while True:
        time_str = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        print(f"{time_str} Batch {i:>6}/{num_total_batches}", end="\r")
        try:
            images, masks, conditional_text_embedding, idx = next(sample_iterator)
        except StopIteration:
            break

        images = images.to(device)
        masks = masks.to(device)
        conditional_text_embedding = conditional_text_embedding.to(device)

        if not args.use_conditional_embeddings or args.dataset_path_embeddings == None:
            conditional_text_embedding = None

        if args.image_retouching_type == "super_resolution":
            images = diffusion.low_res_x(images)
        elif args.image_retouching_type == "colourise":
            images = diffusion.grayscale(images)
        elif args.image_retouching_type == "inpainting":
            images = remove_masked_area(images, masks, device=device, value=args.mask_value)

        if args.conditional_img_add_mask and args.image_retouching_type == "inpainting":
            # append greyscaled mask to conditional image at dimension 1 (dimension for channels)
            masks = T.Grayscale()(masks)
            images = torch.concat((images, masks), dim=1)

        sampled_images = diffusion.sample(
            model, 
            n=images.shape[0], 
            conditional_images=images,
            conditional_text_embedding=conditional_text_embedding,
            batch_size=args.batch_size,
            classifier_free_guidance_scale=guidance_scale
        )

        save_imgs_from_batch(sampled_images, idx, path_output_dir)
        i = i+1

def save_imgs_from_batch(imgs, idx, path_dir):
    for index in range(imgs.shape[0]):
        img = imgs[index]
        img = img.permute(1,2,0).cpu()
        path_img = path_dir / f"{idx[index]}"
        io.imsave(path_img, img)

def generate_imgs_for_dir(path_dir):
    path_chkpt = path_dir / pathlib.Path("ckpt_dict.pt")

    if not path_chkpt.exists():
        print(f"Error: Path {path_chkpt} does not exist!")
        return

    model, train_dataloader, sample_dataloader, mask_text_dataset, diffusion, args, epoch = load_checkpoint(path_chkpt=path_chkpt)

    guidance_scales = [0, 1, 2, 5, 10, 20, 50, 100]

    # if model does not use conditional text/embeddings, then no guidance is possible
    if not args.use_conditional_embeddings or args.dataset_path_embeddings == None:
        guidance_scales = [0]

    for guidance_scale in guidance_scales:
        print(f"Guidance scale: {guidance_scale}")
        generate_imgs_with_model(path_dir, model, sample_dataloader, mask_text_dataset, diffusion, args, guidance_scale=guidance_scale)

if __name__ == "__main__":
    paths = [
        pathlib.Path("test/inpainting/")
    ]
    for path_dir in paths:
        print(f"Dir: {path_dir}")
        generate_imgs_for_dir(path_dir)

