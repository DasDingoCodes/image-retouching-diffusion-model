import os
import torch
import torch.nn as nn
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import logging
from datetime import datetime
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from csv import DictWriter

from unet_extended import UNetExtended
from diffusion import Diffusion
from utils import *


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    setup_logging(args.run_name)
    write_info(args)
    if torch.cuda.device_count() > 1:
        device = args.device
    else:
        device = "cuda"

    image_retouching_types = ["super_resolution", "colourise", "inpainting", None]
    image_retouching_type = args.image_retouching_type
    assert image_retouching_type in image_retouching_types
    use_conditional_image = image_retouching_type != None

    # Classifier free guidance
    use_conditional_embeddings = args.use_conditional_embeddings
    classifier_free_guidance_scale = args.classifier_free_guidance_scale or []
    classifier_free_guidance = classifier_free_guidance_scale is not None
    classifier_free_guidance = classifier_free_guidance and classifier_free_guidance_scale != []
    classifier_free_guidance = classifier_free_guidance and use_conditional_embeddings

    if not classifier_free_guidance:
        classifier_free_guidance_scale = []
    guidance_scales = [0]
    assert type(classifier_free_guidance_scale) == list
    guidance_scales.extend(classifier_free_guidance_scale)
    guidance_scales = sorted(list(set(guidance_scales))) # sort and remove duplicates
    for s in guidance_scales[1:]: # create dirs for sample imgs of different guidance scales (s=0 imgs shall be saved to parent dir)
        os.makedirs(os.path.join("results", args.run_name, f"guidance_{s}"), exist_ok=True)

    mse = nn.MSELoss()
    diffusion = Diffusion(
        img_size=args.image_size, 
        device=device, 
        super_resolution_factor=args.super_resolution_factor,
        noise_steps=args.noise_steps,
        mask_value=args.mask_value,
        classifier_free_guidance=classifier_free_guidance,
        normalised=args.normalise
    )
    
    # 8x8 grid of sample images with fixed random values
    # when saving images, 8 columns are default for grid
    num_sample_imgs = 8*8
    noise_sample = torch.randn((num_sample_imgs, 3, args.image_size, args.image_size)).to(device)
    sample_percentage=0.1
    train_dataloader, sample_dataloader, mask_text_dataset = get_data_img_mask_text(args, sample_percentage=sample_percentage)
    text_embedding_dim = mask_text_dataset.get_embedding_dim()

    # turn off data augmentation for the sample images
    mask_text_dataset.apply_transforms = False
    sample_imgs_from_dataset, sample_masks_from_dataset, sample_embeddings_from_dataset, _ = sample_text_mask_dataset(sample_dataloader, num_sample_imgs)
    sample_imgs_from_dataset = sample_imgs_from_dataset.to(device)
    sample_masks_from_dataset = sample_masks_from_dataset.to(device)
    sample_embeddings_from_dataset = sample_embeddings_from_dataset.to(device)
    mask_text_dataset.apply_transforms = True
    # turn on data augmentation for the training loop

    if not args.use_conditional_embeddings or args.dataset_path_embeddings == None:
        sample_embeddings_from_dataset = None
        use_conditional_embeddings = False

    input_channels = 3
    # if image retouching type is inpainting, then not only the image will be given but the mask as well
    # so the conditional image channels will not be just 3, but 3+1 (greyscaled mask)
    if image_retouching_type == "inpainting" and args.conditional_img_add_mask:
        conditional_img_channels = 4
    else:
        conditional_img_channels = 3
    model = UNetExtended(
        img_shape=(args.image_size, args.image_size),
        input_channels=input_channels,
        hidden=args.hidden,
        num_patches=args.num_patches,
        level_mult = args.level_mult,
        use_self_attention=args.use_self_attention,
        use_conditional_image=use_conditional_image,
        dropout=args.dropout,
        use_conditional_text=use_conditional_embeddings,
        text_embedding_dim=text_embedding_dim,
        device=device,
        output_activation_func=args.model_output_activation_func,
        conditional_img_channels=conditional_img_channels,
        activation_func=args.model_activation_func
    )
    if torch.cuda.device_count() > 1 and len(args.device_ids) > 1:
        model= nn.DataParallel(model,device_ids = args.device_ids)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Save sample images from dataset
    sample_imgs_from_dataset_int = sample_imgs_from_dataset
    if args.normalise:
        sample_imgs_from_dataset_int = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sample_imgs_from_dataset_int)
        sample_imgs_from_dataset_int = sample_imgs_from_dataset_int.clamp(0, 1)
    sample_imgs_from_dataset_int = (sample_imgs_from_dataset_int * 255).type(torch.uint8)
    save_images(sample_imgs_from_dataset_int, os.path.join("results", args.run_name, f"sample_imgs_from_dataset.jpg"))

    # If model gets additional input image, create those inputs from the sample images
    if image_retouching_type == "super_resolution":
        sample_input_imgs = diffusion.low_res_x(sample_imgs_from_dataset)
    elif image_retouching_type == "colourise":
        sample_input_imgs = diffusion.grayscale(sample_imgs_from_dataset)
    elif image_retouching_type == "inpainting":
        sample_input_imgs = remove_masked_area(sample_imgs_from_dataset, sample_masks_from_dataset, device=device, value=args.mask_value)
    else:
        sample_input_imgs = None
    
    # If model gets additional input image, save sample of those inputs as well
    if sample_input_imgs != None:
        sample_input_imgs_int = sample_input_imgs
        if args.normalise:
            sample_input_imgs_int = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sample_input_imgs_int)
            sample_input_imgs_int = sample_input_imgs_int.clamp(0, 1)
        sample_input_imgs_int = (sample_input_imgs_int * 255).type(torch.uint8)
        save_images(sample_input_imgs_int, os.path.join("results", args.run_name, f"sample_input_imgs.jpg"))
    
    if image_retouching_type == "inpainting" and args.conditional_img_add_mask:
        # append greyscaled mask to conditional image at dimension 1 (dimension for channels)
        # do this after saving the conditional image as a file because saving three channels is nicer
        sample_masks_from_dataset = T.Grayscale()(sample_masks_from_dataset)
        sample_input_imgs = torch.concat((sample_input_imgs, sample_masks_from_dataset), dim=1)

    # init FID object
    fid = FrechetInceptionDistance(feature=64, normalize=False, reset_real_features=False)
    fid_input = sample_imgs_from_dataset_int.to("cpu")
    fid.update(fid_input, real=True)

    fid_scores = []
    losses = []

    # ids used in training
    train_data_ids = []
    path_train_data_ids = Path(f'results/{args.run_name}/train_data_ids.txt')

    # avoid displaying matplotlib figures
    plt.ioff()

    path_stats = Path(f'results/{args.run_name}/stats.csv')
    headersCSV = ['MSE','FID']
    with open(path_stats, 'a', newline='') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames=headersCSV)
        dictwriter_object.writeheader()
        f_object.close()


    train_iterator = iter(train_dataloader)
    for epoch in range(args.epochs):
        losses_epoch = 0
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(range(args.steps_per_epoch))
        for i in pbar:
            try:
                images, masks, conditional_text_embedding, idx = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                images, masks, conditional_text_embedding, idx = next(train_iterator)

            train_data_ids.extend(idx)

            images = images.to(device)
            masks = masks.to(device)
            conditional_text_embedding = conditional_text_embedding.to(device)

            if not use_conditional_embeddings:
                conditional_text_embedding = None

            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            if image_retouching_type == "super_resolution":
                conditional_image, d_t, noise = diffusion.super_resolution_noise_data(images, t)
            elif image_retouching_type == "colourise":
                conditional_image, d_t, noise = diffusion.colourise_noise_data(images, t)
            elif image_retouching_type == "inpainting":
                conditional_image, d_t, noise = diffusion.inpainting_noise_data(images, t, masks)
                if args.conditional_img_add_mask:
                    # append greyscaled mask to conditional image at dimension 1 (dimension for channels)
                    masks = T.Grayscale()(masks)
                    conditional_image = torch.concat((conditional_image, masks), dim=1)
            else:
                d_t, noise = diffusion.noise_images(images, t)
                conditional_image = None

            if classifier_free_guidance and conditional_text_embedding is not None:
                # randomly set embeddings to empty tensors
                is_class_cond = torch.rand(size=(images.shape[0], 1), device=device) >= args.classifier_free_guidance_prob
                is_class_cond = is_class_cond.float()
                embedding_mask = expand_like(is_class_cond, conditional_text_embedding)
                conditional_text_embedding = conditional_text_embedding * embedding_mask

            prediction = model(d_t, t, conditional_image=conditional_image, conditional_text_embedding=conditional_text_embedding)
            loss = mse(noise, prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            losses_epoch += loss.item()
        
        # remove duplicate ids and sort list
        train_data_ids = set(train_data_ids)
        train_data_ids = sorted(list(train_data_ids))
        with open(path_train_data_ids, "w") as f:
            f.writelines(train_data_id + "\n" for train_data_id in train_data_ids)
        
        if epoch%args.epochs_per_samples == 0 or epoch == args.epochs -1:
            logging.info(f"Sampling {num_sample_imgs} new images...")

            for guidance_scale in guidance_scales:
                sampled_images = diffusion.sample(
                    model, 
                    n=num_sample_imgs,
                    x=noise_sample,
                    conditional_images=sample_input_imgs,
                    conditional_text_embedding=sample_embeddings_from_dataset,
                    batch_size=args.batch_size,
                    classifier_free_guidance_scale=guidance_scale
                )
                path_sample_imgs = os.path.join("results", args.run_name, f"guidance_{guidance_scale}", f"{epoch}.jpg")
                if guidance_scale == 0:
                    path_sample_imgs = os.path.join("results", args.run_name, f"{epoch}.jpg")
                save_images(sampled_images, path_sample_imgs)
            
            # update fid_scores
            fid.update(sampled_images.to("cpu"), real=False)
            fid_score = fid.compute().item()
            fid.reset()
        
        fid_scores.append(fid_score)
        if args.save_plots:
            path_img = Path(f'results/{args.run_name}/fid_scores.png')
            save_plot(path_img, fid_scores)

        # update losses
        loss = losses_epoch / args.steps_per_epoch
        losses.append(loss)
        if args.save_plots:
            path_img = Path(f'results/{args.run_name}/losses.png')
            save_plot(path_img, losses)

        stats = {"MSE": loss, "FID": fid_score}
        with open(path_stats, 'a', newline='') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=headersCSV)
            dictwriter_object.writerow(stats)
            f_object.close()

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "args": args
        }
        torch.save(checkpoint, os.path.join("results", args.run_name, f"ckpt_dict.pt"))


if __name__ == '__main__':
    _ = torch.manual_seed(1)

    import argparse
    parser = argparse.ArgumentParser()
    time_str = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    args = parser.parse_args()

    args.image_retouching_type = "inpainting" # ["super_resolution", "colourise", "inpainting", None]

    #########################################
    # Training Loop and Sampling Parameters #
    #########################################
    args.epochs = 12
    args.steps_per_epoch = 5
    args.noise_steps = 10
    args.batch_size = 2
    args.dataloader_num_workers = 0
    args.run_name = f"{args.image_retouching_type}"
    args.normalise = False
    args.save_plots = True # loss and FID plots
    args.epochs_per_samples = 5 # how many epochs until intermediate sample gets calulated?
    # the following two are only used if both of them and dataset_path_embeddings are not None
    args.classifier_free_guidance_scale = [1,5,10] # 0 is automatically added
    args.classifier_free_guidance_prob = 0.1

    ####################
    # Model parameters #
    ####################
    args.image_size = 32
    # num_patches is the number of patches on both the width- and height-dimensions.
    # So the image will be divided into num_patches x num_patches 
    args.num_patches = 2
    # level_mult is a list of ints that defines how deep the UNet will be in two ways:
    #  - the i-th of n elements in level_mult represents the i-th DOWN-operation (convolution) 
    #      and the (n-i)-th UP-operation (transposed convolution) in the UNet. 
    #      The longer the list, the more levels the UNet has. 
    #      As each DOWN halves the size of the feature matrix, the maximum number of levels 
    #      depends on image_size and num_patches.
    #  - the value of an element in level_mult scales the number of kernels used in the 
    #      corresponding convolutions / transposed convolutions. 
    #      If the 2nd value is e.g. 16, then the 2nd DOWN-operation will have 16*hidden kernels
    #      and the 2nd to last UP-operation will also have 16*hidden kernels. 
    args.level_mult = [1, 2] 
    # will automatically be set to "cuda" if only one device is available
    args.device = "cuda:2"
    # model can be loaded unto multiple devices. 
    # Will be ignored if there is only one device or if only one device is specified.
    args.device_ids = [2]
    args.lr = 1e-4
    args.hidden = 4
    args.dropout = 0.01
    args.use_self_attention = False
    args.use_conditional_embeddings = True
    args.model_output_activation_func = None # [None, "tanh", "sigmoid", "relu"]
    args.model_activation_func = "relu" # [None, "tanh", "sigmoid", "relu"]

    ######################
    # Dataset Parameters #
    ######################
    # dataset_image_dir
    # examples: 
    #   Path("data/celeba/img_align_celeba") # for all > 200,000 celeba images
    #   Path("data/CelebAMask-HQ/CelebA-HQ-img") # for images with corresponding image masks at Path("data/CelebAMask-HQ/hair_masks")
    args.dataset_image_dir = Path("data/CelebAMask-HQ/CelebA-HQ-img")

    # dataset_mask_dir
    # If given, uses images in this directory as masks for the inpainting task
    # If not given, will create random masks in the images
    # examples:
    #   None
    #   Path("data/CelebAMask-HQ/hair_masks")
    args.dataset_mask_dir = Path("data/CelebAMask-HQ/hair_masks")

    # dataset_path_embeddings can either be path to .pt file, .csv file or path to directory that contains .pt files
    args.dataset_path_embeddings = Path("data/CelebAMask-HQ/attr_hair_revised_first10.csv")

    ###############################
    # Super Resolution Parameters #
    ###############################
    args.super_resolution_factor = 4
    
    ######################### 
    # Inpainting Parameters #
    #########################
    args.conditional_img_add_mask = True # whether to append mask to input image
    args.mask_value = 0.0
    args.inpainting_texts_per_img = None # 10 for CelebAMask-HQ/descriptions_embedded
    train(args)

