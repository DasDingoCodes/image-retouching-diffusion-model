# Image Retouching Diffusion Model

This repository contains a PyTorch based implementation for an image retouching diffusion model. 


## Installation

Create a Conda environment with Python 3.8 and activate it:

    conda create -n YOUR_ENV python=3.8
    conda activate YOUR_ENV

Install PyTorch with Conda:

    conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia

Install rest of requirements with pip:

    pip install -r requirements.txt


## Files

### Scripts

 - `unet_extended.py`: Architecture of the used neural network, based on the U-Net. The network is used to calculate the difference $\mathbf{D}_0$ that shall to be added to the image $\mathbf{Y}$ that shall be changed. To be precise, the network recieves $\mathbf{D}_t$ as an input and outputs $\mathbf{D}_{t-1}$. Additional modules inject the image to-be-changed $\mathbf{Y}$, the conditional embeddings $\mathbf{z}$ (how shall the original image be changed?) and the encoding of the temporal step $t$ into the network. The network can also split the input $\mathbf{D}_t$ and $\mathbf{Y}$ into a quadratic grid of patches.
 - `diffusion.py`: Diffusion class containing the sampling process and functions creating the image-to-be-changed $\mathbf{Y}$ (low resolution, greyscale or masked) and the diffused difference $\mathbf{D}_t$ to the original image at timestep $t$.
 - `utils.py`: Dataset class for loading images, their embeddings and image masks. Also contains some helper functions.
 - `training_loop.py`: Trains neural network with the given arguments. Read the file for its various arguments. Will create a directory with a network checkpoint, intermediate image samples and loss and FID statistics (FID is calculated only on the few intermediately sampled images). 
 - `load_checkpoint.py`: Returns the last saved state of the training process, i.e. neural net, Diffusion class, train and test DataLoader, arguments and current epoch.
 - `generate_original_test_imgs.py`: Takes the test images DataLoader of a loaded checkpoint, creates a directory and fills it with the original test images. These images are the ones that the image retouching diffusion model is supposed to recreate given their respective $\mathbf{Y}$ (low resolution, greyscale or masked images).
 - `generate_sample_imgs.py`: Loads a checkpoint, creates a `test_imgs` folder inside the checkpoint directory and samples the test images. If there are conditional embeddings, the images will be sampled with different Classifier Free Guidance values. For each CFG value a folder will be created inside `test_imgs` (`guidance_scale_0` will always be created), each folder contains the images sampled with the respective CFG value.
 - `calculate_fid.py` and `calculate_ssim.py`: Both files take a directory with the original test images (created by `generate_original_test_imgs.py`) and a checkpoint folder with sampled images (created by `generate_sample_imgs.py`) and calculate for each CFG value the FID/SSIM scores between the original images and the ones sampled with that CFG value. The overall results are just printed out in console. `calculate_ssim.py` additionally writes the similarity scores for each image pair to a file inside the checkpoint directory.

### Data

Some example data based on [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) to demonstrate the diffusion model.

 - `data/CelebAMask-HQ/CelebA-HQ-img`: Contains the first 10 images of CelebAMask-HQ.
 - `data/CelebAMask-HQ/hair_masks`: Contains the hair masks of the first 10 images of CelebAMask-HQ.
 - `data/CelebAMask-HQ/attr.csv`: Contains the attributes of all 30,000 images of CelebAMask-HQ.
 - `data/CelebAMask-HQ/attr_hair_revised.csv`: Contains hair related attributes of all 30,000 images of CelebAMask-HQ. Note that I also took the liberty to annotate the hair colour of 9,303 of the 9,973 images that had no such annotation yet in CelebAMask-HQ.
 - `data/CelebAMask-HQ/attr_hair_revised_first10.csv`: Same as above, but only for the first 10 images of CelebAMask-HQ.