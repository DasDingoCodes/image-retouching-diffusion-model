from pathlib import Path
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImgsDataset(Dataset):
    def __init__(self, path_img_dir):
        self.path_img_dir = path_img_dir
        self.path_imgs = [x for x in self.path_img_dir.iterdir()]
        self.transform = torchvision.transforms.PILToTensor()

    def __len__(self):
        return len(self.path_imgs)
    
    def __get_img__(self, path_img):
        img = Image.open(path_img)
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        img = self.__get_img__(self.path_imgs[idx])
        return img

def fid_update_with_img_dir(fid, path_img_dir, batch_size=64, real_data=False):
    dataset = ImgsDataset(path_img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for i, x in enumerate(dataloader):
        print(f"\t{i}/{len(dataloader)}", end="\r")
        fid.update(x, real=real_data)
    return fid

def df_imgs_calculate_fid(path_orig_dir, paths_result_dirs, batch_size):
    
    paths_result_dirs = [path_result_dir / "test_imgs" for path_result_dir in paths_result_dirs]

    fid = FrechetInceptionDistance(feature=2048, normalize=False, reset_real_features=False)
    fid = fid_update_with_img_dir(fid, path_orig_dir, real_data=True, batch_size=batch_size)

    for path_result_dir in paths_result_dirs:
        print(path_result_dir)
        for path_guidance_scale_dir in path_result_dir.iterdir():
            fid.reset()
            fid = fid_update_with_img_dir(fid, path_guidance_scale_dir, real_data=False, batch_size=batch_size)
            fid_score = fid.compute().item()
            print(f"{path_guidance_scale_dir}:\tFID={fid_score}")

if __name__=="__main__":
    batch_size = 64
    # Image dir and dataframe of original images
    path_orig_dir = Path("data/CelebAMask-HQ_test_imgs")
    # Paths to test_imgs directories.
    # test_imgs directories contain folders for different guidance scales.
    # Each guidance scale folder contains test images sampled with the given guidance scale.
    # The FID will be calculated between the original images dir and the individual guidance scale folders.
    paths_result_dirs = [
        Path("results/inpainting/")
    ]
    df_imgs_calculate_fid(path_orig_dir, paths_result_dirs, batch_size)
