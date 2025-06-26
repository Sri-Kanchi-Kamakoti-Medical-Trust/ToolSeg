import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from einops import rearrange
from monai.networks.utils import one_hot


tools_map = {
    'background': 0,
    'Blade': 1,
    'Cautery': 2,
    'Conjunctivital scissors': 3,
    'Crescent blade': 4,
    'Dialer': 5,
    'Hoskins forceps': 6,
    'Hydrodisection cannula': 7,
    'Keratome': 8,
    'Rhexis needle': 9,
    'Sideport': 10,
    'Simcoe cannula': 11,
    'Vectis': 12,
    'Visco cannula': 13
}

phase_dict = {
    'background': 0,
    'abinjectionandwash': 1,
    'capsulorrhexis': 2,
    'tunnel': 3,
    'conjunctivalcautery': 4,
    'corticalwash': 5,
    'hydroprocedure': 6,
    'incision': 7,
    'mainincisionentry': 8,
    'nucleusdelivery': 9,
    'nucleusprolapse': 10,
    'ovd,iolinsertion': 11,
    'ovdinjection': 12,
    'ovdwash': 13,
    'peritomy': 14,
    'scleralgroove': 15,
    'sideport': 16,
    'stromalhydration': 17,
    'tunnelsuture': 18,
}


class ToolSegDataset(Dataset):

    def __init__(self, data_csv_path, image_dir, mask_dir, mode="train", fold=0, transform=None, phase_condition=None, phase_one_hot=False, eval_mode=False):

        self.data = pd.read_csv(data_csv_path)
        self.data = self.data[self.data[f"fold{fold}"] == mode].reset_index(drop=True)

        self.mode = mode
        self.fold = fold
        self.transform = transform

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.context_dir = "/".join(mask_dir.split("/")[:-1]) + "/img_context"
        self.eval_mode = eval_mode
        self.phase_condition = phase_condition
        self.phase_one_hot = phase_one_hot
        self.num_phases = max(phase_dict.values()) + 1
        self.phase_key = "predicted_phase" if "predicted" in data_csv_path else "phase"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            
        image_path = self.data.loc[idx, "image"]
        mask_path = self.data.loc[idx, "mask"]

        image = Image.open(os.path.join(self.image_dir, image_path))
        image = image.convert("RGB")

        mask = np.load(os.path.join(self.mask_dir, mask_path)).astype(np.uint8)
        mask = rearrange(mask, 'h w c -> c h w')

        image = np.array(image)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).long()

        try:
            phase = phase_dict[self.data.loc[idx, self.phase_key]]
            phase = torch.tensor(phase).long()
        except:
            print("Phase not found:", self.data.loc[idx, self.phase_key])
            phase = torch.tensor(-1).long()

        if self.phase_one_hot:
            #make phase on_hot
            phase = one_hot(phase, num_classes=self.num_phases).reshape(-1)

        if self.eval_mode:
            return image, mask, phase, image_path[:-4]
        return image, mask, phase


if __name__ == "__main__":

    data_csv_path = "SankaraMSICS/cataract-msics-data.csv.csv"
    dataset = ToolSegDataset(data_csv_path, image_dir="./SankaraMSICS/images_4xresized", mask_dir="./SankaraMSICS/masks_4xresized", mode="train", fold=0, transform=None, phase_condition="film", phase_one_hot=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    images, masks, phases = next(iter(dataloader))
    print(images.shape, masks.shape, phases.shape)