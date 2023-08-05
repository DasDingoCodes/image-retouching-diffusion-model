import torch
from torch import optim
from diffusion import Diffusion
from utils import get_data_img_mask_text
from unet_extended import UNetExtended
import torch.nn as nn


def load_checkpoint(path_chkpt):

    # find out device before loading checkpoint because you need to know which device to load variables to
    path_args = path_chkpt.parent / "run_args.txt"
    lines = []
    with open(path_args, "r") as f:
        lines = f.read().splitlines()
    args = {
        x.split(":", 1)[0].strip() : x.split(":", 1)[1].strip() for x in lines
    }
    
    if torch.cuda.device_count() > 1:
        device = args["device"]
    else:
        device = "cuda"

    _ = torch.manual_seed(1)
    checkpoint = torch.load(path_chkpt, map_location=device)
    args = checkpoint["args"]
    epoch = checkpoint["epoch"]
    
    if torch.cuda.device_count() > 1:
        device = args.device
    else:
        device = "cuda"

    image_retouching_type = args.image_retouching_type
    use_conditional_image = image_retouching_type != None

    # Classifier free guidance
    use_conditional_embeddings = args.use_conditional_embeddings
    classifier_free_guidance_scale = args.classifier_free_guidance_scale or []
    classifier_free_guidance = classifier_free_guidance_scale is not None
    classifier_free_guidance = classifier_free_guidance and classifier_free_guidance_scale != []
    classifier_free_guidance = classifier_free_guidance and use_conditional_embeddings

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
    train_dataloader, sample_dataloader, mask_text_dataset = get_data_img_mask_text(args, sample_percentage=0.1)
    text_embedding_dim = mask_text_dataset.get_embedding_dim()

    if not args.use_conditional_embeddings:
        use_conditional_embeddings = False
    
    if "path_embeddings" in vars(args) and args.path_embeddings == None:
        use_conditional_embeddings = False

    if "dataset_path_embeddings" in vars(args) and args.dataset_path_embeddings == None:
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

    model.load_state_dict(checkpoint["model_state"])
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer.load_state_dict(checkpoint["optim_state"])

    return model, train_dataloader, sample_dataloader, mask_text_dataset, diffusion, args, epoch
