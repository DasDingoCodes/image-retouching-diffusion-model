import os
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import pandas as pd
from skimage import io


def save_plot(path, values):
    fig = plt.figure()
    plt.plot(values)
    fig.savefig(path)
    plt.close(fig)

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args, sample_percentage = 0.1):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(Path(args.dataset_path), transform=transforms)
    sample_dataset, train_dataset = random_split(dataset, (sample_percentage, 1.0 - sample_percentage))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    sample_dataloader = DataLoader(sample_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    return train_dataloader, sample_dataloader

def get_fid_init(args, batch_size=32, max_iter: int = None, feature: int = 64, normalize: bool = True):
    """Returns FrechetInceptionDistance object initialised with values of given dataset.
    See: https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html
    
    args.dataset_path: string of path to dataset
    batch_size: batch_size for iterating through dataset, defaults to 32
    max_iter: maximum number of iterations/batches that shall be used for FID calculation. Will run through entire dataset if max_iter = None
    feature: indicates the inceptionv3 feature layer to choose. Can be one of the following: 64, 192, 768, 2048
    normalize: whether to normalise images to [0,1]
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299),  # torchmetrics FID calculation rescales images to 299x299
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.ImageFolder(Path(args.dataset_path), transform=transforms)
    whole_dataset = DataLoader(dataset, batch_size=batch_size)
    fid = FrechetInceptionDistance(feature=feature, normalize=normalize, reset_real_features=False)
    for i, (x, _) in enumerate(whole_dataset):
        if type(max_iter) == int and max_iter <= i:
            break
        print(f"{i}/{len(whole_dataset)}", end="\r")
        fid.update(x, real=True)
    return fid


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def setup_logging(run_name):
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def get_data_img_mask_embeddings(args, sample_percentage):
    # ensure that always the same split for the data is being used
    torch.manual_seed(0)
    path_mask_dir = args.dataset_mask_dir if args.image_retouching_type == "inpainting" else None
    path_embeddings = args.dataset_path_embeddings if args.use_conditional_embeddings else None
    dataset = ImgMaskEmbeddingsDataset(
        path_image_dir=args.dataset_image_dir,
        path_mask_dir=path_mask_dir,
        path_embeddings=path_embeddings,
        embeddings_per_img=args.inpainting_embeddings_per_img,
        img_size=args.image_size,
        normalise=args.normalise
    )
    sample_dataset, train_dataset = random_split(dataset, (sample_percentage, 1.0 - sample_percentage))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    sample_dataloader = DataLoader(sample_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    return train_dataloader, sample_dataloader, dataset

class ImgMaskEmbeddingsDataset(Dataset):
    """Images with masks and descriptions"""

    def __init__(
        self, 
        path_image_dir, 
        path_mask_dir=None, 
        path_embeddings=None, 
        embeddings_per_img=None, 
        img_size=128, 
        normalise=True,
        apply_transforms=True
    ):
        """
        Args:
            path_image_dir: Path of directory containing image files. All images are .jpg files and are named only with an index
            path_mask_dir: Path of directory containing mask files. All masks are .png files and are named only with an index
            path_embeddings: Can be path to a pt file, a csv file, or a path to a directory with the embeddings for the images. Defaults to None.
                pt file: Pytorch tensor should have shape (num_images, embeddings_per_img, embedding_dimensions).
                csv file: for each image there is a row with comma separated values, the first of which being the name of the image file
                directory: For each image there is either a pt file with the embeddings or a directory with #embeddings_per_img .pt embedding files for that image
            embeddings_per_img: How many descriptions there are for each image. 
                If None, will return all embeddings.
                Defaults to None.
            img_size: height and width which the images shall be scaled to, defaults to 128
            normalise: whether to normalise imgs to mean (.5,.5,.5) and std (.5,.5,.5)
        """
        self.path_image_dir = Path(path_image_dir)
        self.path_mask_dir = Path(path_mask_dir) if path_mask_dir else None

        self.apply_transforms=apply_transforms
        
        self.path_embeddings_dir = None
        self.embeddings_in_tensor = False
        self.embeddings_in_dataframe = False
        if path_embeddings != None:
            self.path_embeddings = Path(path_embeddings)

            if self.path_embeddings.is_dir():
                self.path_embeddings_dir = self.path_embeddings
            
            if self.path_embeddings.suffix == ".pt":
                self.embeddings_tensor = torch.load(self.path_embeddings)
                self.embeddings_in_tensor = True

            if self.path_embeddings.suffix == ".csv":
                self.embeddings_df = pd.read_csv(self.path_embeddings)
                self.embeddings_in_dataframe = True

        self.embeddings_per_img = embeddings_per_img
        self.img_size = img_size
        self.normalise = normalise
        # I am checking whether the dataset is the default celeba because the default has a different structure than masked celeba.
        # celeba images have leading 0s and start with 000001 instead of 000000.
        # I don't want to change the dataset on the remote so I will just handle default celeba differently
        self.default_celeba = "celeba" in self.path_image_dir.parts and "img_align_celeba" in self.path_image_dir.parts
        self.default_animefaces = "animefaces" in self.path_image_dir.parts and "original_images" in self.path_image_dir.parts
        if self.default_animefaces:
            self.animefaces_img_paths = [x for x in self.path_image_dir.iterdir()]
    
    def get_embedding_dim(self):
        if self.embeddings_in_tensor:
            return self.embeddings_tensor.shape[1]
        elif self.embeddings_in_dataframe:
            # number of columns -1 because the first column is not an embedding but the name of the image file
            return self.embeddings_df.shape[1] -1
        elif self.path_embeddings_dir != None:
            for p in self.path_embeddings_dir.iterdir():
                single_element = torch.load(p)
                break
            return single_element.shape[1]
        else:
            return None

    def _get_img_path(self, idx):
        """Returns path to image file.
        Do some ugly parsing specifically for default celeba
        I could save all image paths in a list of self.path_image_dir.iterdir() and access it with idx.
        But iterdir() does not guarantee ordered results and the list might take up quite some memory.
        The order for animefaces dataset doesn't matter and it's small, so I use iterdir for it"""
        if self.embeddings_in_dataframe:
            filename = self.embeddings_df.iloc[idx][0]
            return self.path_image_dir / filename
        elif self.default_celeba:
            return self.path_image_dir / f"{idx+1:06d}.jpg"
        elif self.default_animefaces:
            return self.animefaces_img_paths[idx]
        else:
            return self.path_image_dir / f"{idx}.jpg"

    def _get_embeddings(self, idx):
        """Returns embeddings tensor for element with index idx"""
        random_description_index = np.random.randint(self.embeddings_per_img) if self.embeddings_per_img else None
        # if embeddings were given directly as a tensor
        if self.embeddings_in_tensor and random_description_index != None:
            embedded_description = self.embeddings_tensor[idx][random_description_index]
        elif self.embeddings_in_tensor and random_description_index == None:
            embedded_description = self.embeddings_tensor[idx]
        # if embeddings are individual files in a given directory that need to be loaded
        elif self.path_embeddings_dir != None and random_description_index != None:
            path_embedded_description = self.path_embeddings_dir / f"{idx}" / f"{random_description_index}.pt"
            embedded_description = torch.load(path_embedded_description)
        elif self.path_embeddings_dir != None and random_description_index == None:
            path_embedded_description = self.path_embeddings_dir / f"{idx}.pt"
            embedded_description = torch.load(path_embedded_description)
        # if embeddings are in dataframe
        elif self.embeddings_in_dataframe:
            row = self.embeddings_df.iloc[idx]
            # first column is filename, not part of embedding
            embedded_description = torch.Tensor(row[1:])
        else:
            embedded_description = torch.empty(1)
        return embedded_description

    def _get_mask(self, idx, image):
        """Returns mask for image"""
        if self.path_mask_dir == None:
            # Create random mask if no mask dir given
            mask = torch.zeros_like(image)
            mask = torchvision.transforms.RandomErasing(p=0.9,value=1.0)(mask)
            return mask

        if self.embeddings_in_dataframe:
            filename = self.embeddings_df.iloc[idx][0]
            filename = Path(filename).stem
        else:
            filename = idx
        
        path_mask = self.path_mask_dir / f"{filename}.png"
        mask = io.imread(path_mask)
        mask = torchvision.transforms.ToTensor()(mask)
        return mask

    def grow_mask(self, mask):
        """Grows masked area in given mask by 1 pixel"""
        new_mask = mask.detach().clone()
        C,H,W = mask.shape
        for i in range(H):
            for j in range(W):
                if new_mask[0,i,j] == 1.0:
                    continue
                # above
                if i > 0 and mask[0, i-1, j] > 0:
                    new_mask[:,i,j] = 1.0
                    continue
                # under
                if i < H-1 and mask[0, i+1, j] > 0:
                    new_mask[:,i,j] = 1.0
                    continue
                # left
                if j > 0 and mask[0, i, j-1] > 0:
                    new_mask[:,i,j] = 1.0
                    continue
                # left
                if j < W-1 and mask[0, i, j+1] > 0:
                    new_mask[:,i,j] = 1.0
                    continue
        return new_mask

    def shrink_mask(self, mask):
        """Shrinks masked area in given mask by 1 pixel"""
        new_mask = mask.detach().clone()
        C,H,W = mask.shape
        for i in range(H):
            for j in range(W):
                if new_mask[0,i,j] == 0.0:
                    continue
                # above
                if i > 0 and mask[0, i-1, j] == 0:
                    new_mask[:,i,j] = 0.0
                    continue
                # under
                if i < H-1 and mask[0, i+1, j] == 0:
                    new_mask[:,i,j] = 0.0
                    continue
                # left
                if j > 0 and mask[0, i, j-1] == 0:
                    new_mask[:,i,j] = 0.0
                    continue
                # left
                if j < W-1 and mask[0, i, j+1] == 0:
                    new_mask[:,i,j] = 0.0
                    continue
        return new_mask

    def _get_image(self, idx):
        """Returns image for given index"""
        path_img = self._get_img_path(idx)
        image = io.imread(path_img)
        image = torchvision.transforms.ToTensor()(image)
        return image

    def __len__(self):
        if self.embeddings_in_dataframe:
            return self.embeddings_df.shape[0]
        return len([x for x in self.path_image_dir.iterdir()])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self._get_image(idx)
        mask = self._get_mask(idx, image)
        embedded_description = self._get_embeddings(idx)
        img_name = self._get_img_path(idx).name

        
        if not self.apply_transforms:
            resize = torchvision.transforms.Resize(size=(self.img_size, self.img_size))
            image = resize(image)
            mask = resize(mask)
            return image, mask, embedded_description, img_name
        
        # Colour jitter
        jitter = torchvision.transforms.ColorJitter(brightness=.25, hue=0.025, saturation=0.25)
        image = jitter(image)

        # Resize 
        resize = torchvision.transforms.Resize(size=(self.img_size + 10, self.img_size + 10))
        image = resize(image)
        mask = resize(mask)

        # Set all values > 0 to 1 in mask
        mask = mask != 0
        mask = mask.float()

        # Random mask growing/shrinking if mask was not randomly generated
        if self.path_mask_dir is not None:
            rand = np.random.random()
            if rand < 0.9:
                pass
            elif rand < 0.975:
                mask = self.grow_mask(mask)
            else:
                mask = self.shrink_mask(mask)

        # Random horizontal flipping
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Rotate
        rotation = torchvision.transforms.RandomRotation.get_params(degrees=[-30,30])
        image = TF.rotate(image, rotation)
        mask = TF.rotate(mask, rotation)

        # Random crop
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            image, output_size=(self.img_size, self.img_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Normalise
        if self.normalise:
            image = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        return image, mask, embedded_description, img_name

def sample_img_mask_embeddings_dataset(dataloader, num_sample_imgs):
    iterator = iter(dataloader)
    sample_imgs_from_dataset, sample_masks_from_dataset, sample_embeddings_from_dataset, img_names = next(iterator)
    img_names = list(img_names)
    while sample_imgs_from_dataset.shape[0] < num_sample_imgs:
        try:
            imgs_next_batch, masks_next_batch, embeddings_next_batch, img_names_next_batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            imgs_next_batch, masks_next_batch, embeddings_next_batch, img_names_next_batch = next(iterator)
        sample_imgs_from_dataset = torch.concat((sample_imgs_from_dataset, imgs_next_batch))
        sample_masks_from_dataset = torch.concat((sample_masks_from_dataset, masks_next_batch))
        sample_embeddings_from_dataset = torch.concat((sample_embeddings_from_dataset, embeddings_next_batch))
        img_names.extend(img_names_next_batch)
    sample_masks_from_dataset = sample_masks_from_dataset[:num_sample_imgs]
    sample_embeddings_from_dataset = sample_embeddings_from_dataset[:num_sample_imgs]
    sample_imgs_from_dataset = sample_imgs_from_dataset[:num_sample_imgs]

    return sample_imgs_from_dataset, sample_masks_from_dataset, sample_embeddings_from_dataset, img_names

def remove_masked_area(images: torch.Tensor, masks: torch.Tensor, value=None, mean=0.5, std=0.5, device="cpu"):
    """Replaces areas in images covered by masks with either noise or -if given- a constant value."""
    imgs_masks_removed = images * (1 - masks)
    if value == 0.0:
        replacement_values = torch.zeros_like(masks).to(device)
    elif value != None:
        replacement_values = torch.ones_like(masks).to(device)
        if value != 1.0:
            replacement_values *= value
    else:
        replacement_values = torch.normal(size=masks.size(), mean=mean, std=std).to(device)
    return imgs_masks_removed + masks * replacement_values

def expand_like(expandable: torch.Tensor, target: torch.Tensor):
    """Expands first tensor to the shape of the second tensor, returns result"""
    return expandable.view(*expandable.shape, *(1,)*(target.ndim-expandable.ndim)).expand_as(target)

def write_info(args):
    filename = os.path.join("results", args.run_name, "run_args.txt")
    with open(filename, "a") as f:
        for arg in vars(args):
            row = f"{arg}: {getattr(args, arg)}\n"
            f.write(row)
