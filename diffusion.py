import torch
import torchvision.transforms as T

from utils import *


class Diffusion:
    def __init__(
        self, 
        noise_steps=1000, 
        beta_start=1e-4, 
        beta_end=0.02, 
        img_size=256, 
        device="cuda", 
        super_resolution_factor=4, 
        mask_value=None,
        classifier_free_guidance=False,
        normalised=False
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.super_resolution_factor = super_resolution_factor
        self.mask_value = mask_value
        self.classifier_free_guidance = classifier_free_guidance
        self.normalised = normalised

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def super_resolution_noise_data(self, x, t):
        """Returns x_L (low resolution x), d_t (diffused difference between x and x_L at timestep t) and Ɛ (noise inserted into d_t at timestep t)"""
        x_L = self.low_res_x(x)
        d_t, Ɛ = self.noise_images(x - x_L, t)
        return x_L, d_t, Ɛ
    
    def colourise_noise_data(self, x, t):
        """Returns x_g (grayscaled x), d_t (diffused difference between x and x_g at timestep t) and Ɛ (noise inserted into d_t at timestep t)"""
        x_g = self.grayscale(x, num_output_channels=3)
        d_t, Ɛ = self.noise_images(x - x_g, t)
        return x_g, d_t, Ɛ

    def inpainting_noise_data(self, x, t, mask):
        """Returns x_masked (x with masked area removed), d_t (diffused difference between x and x_masked at timestep t) and Ɛ (noise inserted into d_t at timestep t)"""
        x_masked = remove_masked_area(x, mask, device=self.device, value=self.mask_value)
        d_t, Ɛ = self.noise_images(x - x_masked, t)
        return x_masked, d_t, Ɛ


    def low_res_x(self, x):
        x_L = T.Resize(size=self.img_size//self.super_resolution_factor)(x)
        x_L = T.Resize(size=self.img_size)(x_L)
        return x_L

    def grayscale(self, x, num_output_channels=3):
        return T.Grayscale(num_output_channels=num_output_channels)(x)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, x=None, conditional_images=None, conditional_text_embedding=None, batch_size=None, classifier_free_guidance_scale=0):
        if batch_size==None:
            batch_size=n
        model.eval()
        with torch.no_grad():
            if x == None:
                x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            prediction = None
            for i in range(0,n,batch_size):
                x_batch = x[i:i+batch_size]
                if conditional_images is None:
                    conditional_images_batch = None
                else:
                    conditional_images_batch = conditional_images[i:i+batch_size]
                if conditional_text_embedding is None:
                    conditional_text_embedding_batch = None
                else:
                    conditional_text_embedding_batch = conditional_text_embedding[i:i+batch_size]
                batch_prediction = self._batch_predict(
                    model,
                    x=x_batch, 
                    conditional_images=conditional_images_batch, 
                    conditional_text_embedding=conditional_text_embedding_batch,
                    classifier_free_guidance_scale=classifier_free_guidance_scale
                )
                if isinstance(prediction, type(None)):
                    prediction = batch_prediction
                else:
                    prediction = torch.concat((prediction, batch_prediction))
            
            x = prediction

        model.train()
        if conditional_images != None:
            # if image retouching, then model predicts difference between original_imgs and wanted images
            # in that case, add original_imgs to the output of the model to get a prediction for the wanted image
            # if inpainting, then the 4th channel of conditional images contains the masks
            # consider only the first three channels for the addition 
            x += conditional_images[:,0:3,:,:]
        x = x.clamp(0, 1)
        x = (x * 255).type(torch.uint8)
        return x

    def _batch_predict(self, model, x=None, conditional_images=None, conditional_text_embedding=None, classifier_free_guidance_scale=0):
        batch_size = x.shape[0]
        for i in reversed(range(1, self.noise_steps)):
            t = (torch.ones(batch_size) * i).long().to(self.device)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            prediction = model(x, t, conditional_image=conditional_images, conditional_text_embedding=conditional_text_embedding)

            if self.classifier_free_guidance and classifier_free_guidance_scale > 0:
                empty_embeddings = torch.zeros_like(conditional_text_embedding)
                prediction_unconditional = model(x,t, conditional_image=conditional_images, conditional_text_embedding=empty_embeddings)
                prediction = (1 + classifier_free_guidance_scale) * prediction
                prediction = prediction - classifier_free_guidance_scale * prediction_unconditional
            
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * prediction) + torch.sqrt(beta) * noise
        
        return x
