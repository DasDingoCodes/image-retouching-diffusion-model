# Image Retouching Diffusion Model

This repository contains a PyTorch based implementation for training an image retouching diffusion model from scratch. 
The following image-to-image tasks are supported:

- Inpainting
- Super Resolution
- Colourisation


## Installation

Create a Conda environment with Python 3.8 and activate it:

    conda create -n YOUR_ENV python=3.8
    conda activate YOUR_ENV

Install PyTorch with Conda:

    conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia

Install rest of requirements with pip:

    pip install -r requirements.txt

## Usage

With the following steps you can train a diffusion model for inpainting.
Read the script files for further information on how to run them.

Set up some training data, for example [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ):

  - Put the original images into a folder, e.g. `data/CelebAMask-HQ/CelebA-HQ-img`.
  - Put the corresponding .png mask images into another folder, e.g. `data/CelebAMask-HQ/hair_masks`. If there is an original image `1.jpg`, then there must be a mask image `1.png`.
  - Add a .csv file containing the image names and attributes for that image, e.g. `data/attr.csv`. The first value of a row must be the name of an original image, e.g. `1.jpg`, the rest numerical values.

Activate the Conda environment you created above:

    conda activate YOUR_ENV

Train a model for 12 epochs on the tiny dataset:

    python training_loop.py

This creates the `results/inpainting` directory with a model checkpoint and sample images at some epochs. Next we will create the original images of the test data, i.e. the images we want to reconstruct, and afterwards the reconstructions of these test images:

    python generate_original_test_imgs.py
    python generate_sample_imgs.py

Now we can compare the original images and the reconstructions. The following commands will print out the FID and SSIM scores of the test images:

    python calculate_fid.py
    python calculate_ssim.py



## Files

 - `unet_extended.py`: Architecture of the used neural network, based on the U-Net. The network is used for the calculation of the difference $\mathbf{D}_0$ that will be added to the image $\mathbf{Y}$ which shall be changed. To be precise, the network recieves $\mathbf{D}\_t$ as an input and outputs $\mathbf{D}\_{t-1}$. Additional modules inject the image to-be-changed $\mathbf{Y}$, the conditional embeddings $\mathbf{z}$ (how shall the original image be changed?) and the encoding of the temporal step $t$ into the network. The network can also split the input $\mathbf{D}_t$ and $\mathbf{Y}$ into a quadratic grid of patches.
 - `diffusion.py`: Diffusion class containing the sampling process and functions creating the image-to-be-changed $\mathbf{Y}$ (low resolution, greyscale or masked) and the diffused difference $\mathbf{D}_t$ to the original image at timestep $t$.
 - `utils.py`: Dataset class for loading images, their embeddings and image masks. Also contains some helper functions.
 - `training_loop.py`: Trains neural network with the given arguments. Read the file for its various arguments. Will create a directory with a network checkpoint, intermediate image samples and loss and FID statistics (FID is calculated only on the few intermediately sampled images). 
 - `load_checkpoint.py`: Returns the last saved state of the training process, i.e. neural net, Diffusion class, train and test DataLoader, arguments and current epoch.
 - `generate_original_test_imgs.py`: Takes the test images DataLoader of a loaded checkpoint, creates a directory and fills it with the original test images. These images are the ones that the image retouching diffusion model is supposed to recreate given their respective $\mathbf{Y}$ (low resolution, greyscale or masked images).
 - `generate_sample_imgs.py`: Loads a checkpoint, creates a `test_imgs` folder inside the checkpoint directory and samples the test images. If there are conditional embeddings, the images will be sampled with different Classifier Free Guidance values. For each CFG value a folder will be created inside `test_imgs` (`guidance_scale_0` will always be created), each folder contains the images sampled with the respective CFG value.
 - `calculate_fid.py` and `calculate_ssim.py`: Both files take a directory with the original test images (created by `generate_original_test_imgs.py`) and a checkpoint folder with sampled images (created by `generate_sample_imgs.py`) and calculate for each CFG value the FID/SSIM scores between the original images and the ones sampled with that CFG value. The overall results are just printed out in console. `calculate_ssim.py` additionally writes the similarity scores for each image pair to a file inside the checkpoint directory.