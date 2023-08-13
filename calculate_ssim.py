import json
from pathlib import Path
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class RealReconstructedImgDataset(Dataset):
    def __init__(self, path_real_img_dir, path_reconstructed_img_dir):
        self.path_real_img_dir = path_real_img_dir
        self.path_reconstructed_img_dir = path_reconstructed_img_dir
        self.img_names = [img_path.name for img_path in self.path_real_img_dir.iterdir()]
        self.transform = torchvision.transforms.PILToTensor()

    def __len__(self):
        return len(self.img_names)
    
    def __get_img__(self, path_img, img_size=None):
        img = Image.open(path_img)
        if img_size is not None:
            img = img.resize(img_size)
        img = self.transform(img) / 255.0
        return img

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        reconstructed_img =  self.__get_img__(self.path_reconstructed_img_dir / img_name)
        _, img_width, img_height = reconstructed_img.shape
        real_img = self.__get_img__(self.path_real_img_dir / img_name, img_size=(img_width, img_height))

        return real_img, reconstructed_img, img_name

def ssim_scores_for_img_dirs(path_real_img_dir, path_fake_img_dir, **kwargs):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None, **kwargs)
    eval_dataset = RealReconstructedImgDataset(path_real_img_dir, path_fake_img_dir)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64)
    ssim_scores = []
    ssim_scores_dict = {}
    for i, (real, reconstructed, img_names) in enumerate(eval_dataloader):
        print(f"{i}/{len(eval_dataloader)}", end="\r")
        ssim_score_batch = ssim(real, reconstructed)
        ssim_scores_dict.update({img_name:ssim_score.item() for (img_name, ssim_score) in zip(img_names, ssim_score_batch)})
        ssim_scores.extend([ssim_score.item() for ssim_score in ssim_score_batch])
    return ssim_scores, ssim_scores_dict

def ssim_scores_for_result_dirs(path_orig_dir, path_result_dirs, ssim_kernel_size=11):

    path_result_dirs = [path_result_dir / "test_imgs" for path_result_dir in path_result_dirs]

    for path_result_dir in path_result_dirs:
        print(path_result_dir)
        for path_guidance_scale_dir in path_result_dir.iterdir():
            ssim_scores, ssim_scores_dict = ssim_scores_for_img_dirs(path_orig_dir, path_guidance_scale_dir, kernel_size=ssim_kernel_size)
            ssim_score = np.mean(ssim_scores)
            ssim_score_std = np.std(ssim_scores)
            print(f"{path_guidance_scale_dir}:\tSSIM={ssim_score} (+/- {ssim_score_std})")

            path_ssim_scores = path_guidance_scale_dir.parent.parent / f"ssim_scores_{path_guidance_scale_dir.name}.json"
            with open(path_ssim_scores, 'w', encoding='utf-8') as f:
                json.dump(ssim_scores_dict, f, ensure_ascii=False, indent=4)


if __name__=="__main__":
    batch_size = 64

    path_orig_dir = Path("data/CelebAMask-HQ_test_imgs")

    paths_result_dirs = [
        Path("results/inpainting/")
    ]

    # default is 11
    ssim_kernel_size = 11

    ssim_scores_for_result_dirs(path_orig_dir, paths_result_dirs, ssim_kernel_size=ssim_kernel_size)