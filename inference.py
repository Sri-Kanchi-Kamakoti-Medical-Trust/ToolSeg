import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm
from matplotlib import pyplot as plt

from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from monai.data import decollate_batch

from models import UNet, UNetPCD, PAANet, UNetGatedPCD, TernausNet11
from dataset import ToolSegDataset
from utils import get_config, set_seed


config = get_config("config.yaml")

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
instruments = ["background"] + config["dataset"]["instrument_labels"]
num_classes = len(instruments)
phase_dict = {
    'background': 0,
    'abinjectionandwash': 1,
    'capsulorrhexis': 2,
    'cautery': 3,
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
    'tunnel': 18,
    'tunnelsuture': 19
}

color_map = {
        0: [0, 0, 0],       # Black background
        1: [255, 0, 0],     # Red
        2: [0, 255, 0],     # Green
        3: [0, 0, 255],     # Blue
        4: [255, 255, 0],   # Yellow
        5: [255, 0, 255],   # Magenta
        6: [0, 255, 255],   # Cyan
        7: [128, 128, 128], # Gray
        8: [128, 0, 0],     # Maroon
        9: [0, 128, 0],     # Dark Green
        10: [0, 0, 128],    # Navy
        11: [128, 128, 0],  # Olive
        12: [128, 0, 128],  # Purple
        13: [0, 128, 128]   # Teal
}


def convert_to_rgb(segmentation_array):
    
    h, w = segmentation_array.shape
    
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for key, color in color_map.items():
        rgb_image[segmentation_array == key] = color
    
    return rgb_image


def generate_output(model, loader, dice_metric, dice_metric_batch, iou_metric, iou_metric_batch, device, post_transform, output_dir, split):

    if not os.path.exists(os.path.join(output_dir, split)):
            os.makedirs(os.path.join(output_dir, split))

    output_dir = os.path.join(output_dir, split)
    
    with torch.no_grad():
        for i, (image, mask, phase, image_name) in enumerate(tqdm(loader)):
            image = image.to(device)
            mask = mask.to(device)
            phase = phase.to(device)

            image_name = image_name[0]

            if config['condition']['phase'] == 'none':
                output = model(image)
            else:
                output = model(image, phase)

            # squeeze the batch dimension
            # output = output.squeeze(0)
            # mask = mask.squeeze(0)

            # output = post_transform(output)
            # output = torch.argmax(output, dim=1, keepdim=True)
            masks_pred = [post_transform(mask_pred) for mask_pred in decollate_batch(output)]
            output = torch.stack(masks_pred, dim=0)

            dice_metric(y_pred=output, y=mask)
            dice_metric_batch(y_pred=output, y=mask)
            #make mask one hot
            one_hot_mask = torch.nn.functional.one_hot(mask.view(1, mask.shape[-2], mask.shape[-1]), num_classes=num_classes).permute(0, 3, 1, 2)
            iou_metric(y_pred=output, y=one_hot_mask)
            
            for tool in np.unique(mask.cpu().numpy()):
                if tool == 0:
                    continue
                one_hot_mask_tool = one_hot_mask[:, tool, ...]
                output_tool = output[:, tool, ...]
                iou_metric_batch[tool](y_pred=output_tool, y=one_hot_mask_tool)

            # visualize multi-class output
            output = output.cpu().numpy()
            mask = mask.cpu().numpy()
            phase = phase.cpu().numpy().argmax(axis=1)

            # convert the numpy array to rgb mask
            output = convert_to_rgb(np.squeeze(AsDiscrete(argmax=True)(np.squeeze(output))))
            mask = convert_to_rgb(np.squeeze(mask))

            output = Image.fromarray(output)
            mask = Image.fromarray(mask)

            # output.save(os.path.join(output_dir, f"{image_name}_output.png"))
            # mask.save(os.path.join(output_dir, f"{image_name}_mask.png"))

            # concatenate output and mask and plot them as subplots and save as a single image using matplotlib
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))

            image = Image.open(os.path.join(config["dataset"]["image_dir"], f"{image_name}.png"))
            ax[0].imshow(image)
            ax[0].set_title("Image")
            ax[0].axis("off")

            ax[1].imshow(output)
            ax[1].set_title("Output")
            ax[1].axis("off")

            ax[2].imshow(mask)
            ax[2].set_title("Mask")
            ax[2].axis("off")
            
            # add color_map as a legend outside the plot - not colour map are in 0-255 range
            normalized_color_map = {v: [i/255 for i in color_map[v]] for v in color_map}
            # get the label for each color in the color map from the tools_map
            labels = {v: k for k, v in tools_map.items()}
            # add the legend to the plot outside the two subplots
            fig.legend(handles=[plt.Line2D([0], [0], color=normalized_color_map[v], label=labels[v]) for v in normalized_color_map], loc="center left", borderaxespad=0.1, title="Legend", fontsize=7)
            # fig.subplots_adjust(right=0.8)
            # give title to the figure with fold name and image name and the phase condition along with the phase name based on the phase_dict
            fig.suptitle(f"fold: {config['fold']} - image: {image_name} - phase: {list(phase_dict.keys())[list(phase_dict.values()).index(phase)]} - condition: {config['condition']['phase']}", fontsize=16)

            plt.savefig(os.path.join(output_dir, f"{image_name}.png"))
            plt.close()

            # if i == 1:
            #     break

    dice_classwise_score = dice_metric_batch.aggregate()
    # dice_mean_score = dice_metric.aggregate().item()
    dice_mean_score = dice_metric.aggregate().item()

    print(f"{split}_dice_score: {dice_mean_score:.4f}")

    for key in dice_classwise_score:
        print(f"{key}: {dice_classwise_score[key]}")

    print(f"The average dice score is: {np.mean(list(dice_classwise_score.values())):.4f}")

    dice_metric.reset()
    dice_metric_batch.reset()

    iou_mean_score = iou_metric.aggregate().item()
    iou_classwise_score = {
    }
    for i in range(1, num_classes):
        try:
            iou_classwise_score[instruments[i]] = iou_metric_batch[i].aggregate().item()
        except:
            print(f"Error in class: {i}")
            iou_classwise_score[instruments[i]] = 0

    print(f"{split}_iou_score: {iou_mean_score:.4f}")

    for key in iou_classwise_score:
        print(f"{key}: {iou_classwise_score[key]}")

    print(f"The average iou score is: {np.mean(list(iou_classwise_score.values())):.4f}")
    
    #create a dataframe with these numbers and save it as a csv file
    data = [
        dice_classwise_score,
        iou_classwise_score,
    ]
    df = pd.DataFrame(data, index=["dice", "iou"])
    df['model'] = config['model']['name']
    df['fold'] = config['fold']
    df['condition'] = config['condition']['phase'] + ('_predicted' if 'predicted' in config['weights'] else '')
    df['mean'] = [
        np.mean(list(dice_classwise_score.values())),
        np.mean(list(iou_classwise_score.values()))
    ]
    #place model, fold, mean at the beginning of the dataframe
    cols = df.columns.tolist()
    cols = cols[-4:] + cols[:-4]
    df = df[cols]

    df.to_csv(os.path.join(output_dir, f"{config['model']['name']}_{config['fold']}_{split}_scores.csv"))


def inference():
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False,
                            return_with_label=config["dataset"]["instrument_labels"])
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    classwise_iou_metric = [  MeanIoU(include_background=False, reduction="mean") for _ in range(num_classes) ]
    
    post_transform = AsDiscrete(argmax=True, to_onehot=config['model']['n_classes'])
    print(config['model'])
    if config['model']['name'] == "unet":
        if config["condition"]["phase"] == "none":
            model = UNet(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"]).to(device)
        elif config["condition"]["phase"] == "pcd":
            model = UNetPCD(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"], n_phases=config["model"]["n_phases"]).to(device)
        elif config["condition"]["phase"] == "pcd-gated":
            model = UNetGatedPCD(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"], n_phases=config["model"]["n_phases"]).to(device)
    elif config['model']['name'] == "paanet":
        model = PAANet(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"]).to(device)
    
    # load the last saved best model based on validation metric
    weights_path = config['weights']
    weights = "model_best_val_dsc"

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(weights_path, f"{weights}.pth"))['model_state_dict'])
    model.eval()

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    experiment_name = weights_path.strip('/').split('/')[-1]
    output_dir_path = os.path.join(config['output_dir'], experiment_name)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    train_transforms, val_transforms, test_transforms = None, None, None

    test_dataset = ToolSegDataset(config["dataset"]["data_csv_path"], image_dir=config["dataset"]["image_dir"], mask_dir=config["dataset"]["mask_dir"], mode="test", fold=config["fold"], transform=test_transforms, phase_condition=config["condition"]["phase"], phase_one_hot=config["dataset"]["phase_one_hot"], eval_mode=True)
    test_loader = DataLoader(test_dataset, num_workers=config["num_workers"], pin_memory=True)

    generate_output(model, test_loader, dice_metric, dice_metric_batch, iou_metric, classwise_iou_metric, device, post_transform, output_dir_path, split="test")

if __name__ == '__main__':
    set_seed(config["seed"])
    inference()