import os
import wandb
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import albumentations as A

from monai.losses import DiceCELoss, Dice


from models import UNet, UNetPCD, UNetGatedPCD, PAANet, TernausNet11
from dataset import ToolSegDataset
from utils import get_config, set_seed
from train import train


config = get_config("config.yaml")



def main(run_name):

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_transforms = A.Compose([
    #     A.Downscale(scale_range=(0.25, 0.25), always_apply=True, p=1.0),
    # ], is_check_shapes=False)
    # val_transforms = A.Compose([
    #     A.Downscale(scale_range=(0.25, 0.25), always_apply=True, p=1.0),
    # ], is_check_shapes=False)
    # test_transforms = A.Compose([
    #     A.Downscale(scale_range=(0.25, 0.25), always_apply=True, p=1.0),
    # ], is_check_shapes=False)
    train_transforms, val_transforms, test_transforms = None, None, None

    # dataset
    train_dataset = ToolSegDataset(config["dataset"]["data_csv_path"], image_dir=config["dataset"]["image_dir"], mask_dir=config["dataset"]["mask_dir"], mode="train", fold=config["fold"], transform=train_transforms, phase_condition=config["condition"]["phase"], phase_one_hot=config["dataset"]["phase_one_hot"])
    val_dataset = ToolSegDataset(config["dataset"]["data_csv_path"], image_dir=config["dataset"]["image_dir"], mask_dir=config["dataset"]["mask_dir"], mode="validation", fold=config["fold"], transform=val_transforms, phase_condition=config["condition"]["phase"], phase_one_hot=config["dataset"]["phase_one_hot"])
    test_dataset = ToolSegDataset(config["dataset"]["data_csv_path"], image_dir=config["dataset"]["image_dir"], mask_dir=config["dataset"]["mask_dir"], mode="test", fold=config["fold"], transform=test_transforms, phase_condition=config["condition"]["phase"], phase_one_hot=config["dataset"]["phase_one_hot"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True)

    # model
    if config['model']['name'] == "unet":
        if config["condition"]["phase"] == "none":
            model = UNet(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"]).to(device)
        elif config["condition"]["phase"] == "pcd":
            model = UNetPCD(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"], n_phases=config["model"]["n_phases"]).to(device)
        elif config["condition"]["phase"] == "pcd-gated":
            model = UNetGatedPCD(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"], n_phases=config["model"]["n_phases"]).to(device)
    elif config['model']['name'] == "paanet":
        model = PAANet(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"]).to(device)
    elif config['model']['name'] == "ternausnet":
        model = TernausNet11(num_classes=config["model"]["n_classes"], num_filters=32, pretrained=True).to(device)
    model = nn.DataParallel(model)

    # loss and optimizer
    criterion = DiceCELoss(softmax=True, to_onehot_y=True)
    optimizer = optim.Adam(model.parameters(), lr=config["init_lr"])

    save_dir = os.path.join(config["save_dir"], run_name)
    if not os.path.exists(save_dir):
       os.makedirs(save_dir)

    if config['pretrained']:
        model.load_state_dict(torch.load(config["pretrained_model_ckpt"], map_location=device)["model_state_dict"])

    # train, eval and test
    train(model=model, config=config, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, optimizer=optimizer, criterion=criterion, save_dir=save_dir, n_epochs=config["n_epochs"], device=device)


if __name__ == "__main__":
    
    project_name = config["project_name"]
    pseudo_str = '_pseudo' if 'pseudo' in config['dataset']['data_csv_path'] else ''
    predicted_str = '_predicted_phase' if 'predicted' in config['dataset']['data_csv_path'] else ''
    pretrain_str = '_pretrained' if config['pretrained'] else ''
    run_name = f"{config['model']['name']}{pseudo_str}{predicted_str}{pretrain_str}_fold_{config['fold']}_phase_{config['condition']['phase']}_dataaug_{config['data_aug']}_nchannel_{config['model']['n_channels']}_lr_{config['init_lr']}_bs_{config['batch_size']}_epochs_{config['n_epochs']}"
    run_name = run_name + "_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    wandb.init(
        project=project_name,
        config=config,
        name=run_name,
        mode="disabled"    
    )

    wandb.config.update(config)

    if not os.path.exists(config["save_dir"]) and not config['is_debug']:
        os.makedirs(config["save_dir"])

    save_dir = os.path.join(config["save_dir"], run_name)
    os.makedirs(save_dir)

    main(run_name=run_name)

    wandb.finish()