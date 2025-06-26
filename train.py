import torch

import time
import wandb
import numpy as np
from tqdm import tqdm

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from test import test
from utils import load_checkpoint, save_checkpoint, EarlyStopper



def train(model, config, train_loader, val_loader, test_loader, optimizer, criterion, save_dir, device, n_epochs=10):

    best_val_dsc = -1
    best_val_dsc_epoch = -1

    best_val_loss = float('inf')
    best_val_loss_epoch = -1

    early_stopper = EarlyStopper(patience=config['patience'], min_delta=0)
    
    for epoch in range(n_epochs):

        # train_epoch and eval_epoch functions
        train_loss, train_dsc, train_classwise_dsc = train_epoch(train_loader, config, model, optimizer, criterion, device)
        val_loss, val_dsc, val_classwise_dsc = eval_epoch(val_loader, config, model, criterion, device)

        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            best_val_dsc_epoch = epoch + 1

            if not config['is_debug']:
                save_checkpoint(model, optimizer, save_dir, epoch+1)
                save_checkpoint(model, optimizer, save_dir, "best_val_dsc")
                print(f"New best DSC metric checkpoint saved at {save_dir}model_{epoch+1}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch + 1

            if not config['is_debug']:
                save_checkpoint(model, optimizer, save_dir, epoch+1)
                save_checkpoint(model, optimizer, save_dir, "best_val_loss")
                print(f"New best loss metric checkpoint saved at {save_dir}model_{epoch+1}.pth")

        print("Current epoch: {} current mean val dice: {:.4f} best mean val dice: {:.4f} at epoch {}, current val loss: {:.4f} best val loss: {:.4f} at epoch {}".format(epoch + 1, val_dsc, best_val_dsc, best_val_dsc_epoch, val_loss, best_val_loss, best_val_loss_epoch))
        
        test_loss, test_dsc, test_classwise_dsc = test(test_loader, config, model, criterion, device)

        # log wandb
        wandb.log({"train/loss": train_loss, "train/dice_score": train_dsc, "val/loss": val_loss, "val/dice_score": val_dsc, "test/loss": test_loss, "test/dice_score": test_dsc, "epoch": epoch+1, "best_val_dice_score": best_val_dsc, "best_val_dice_score_epoch": best_val_dsc_epoch, "train/classwise_dice_score": train_classwise_dsc, "val/classwise_dice_score": val_classwise_dsc, "test/classwise_dice_score": test_classwise_dsc, "best_val_loss": best_val_loss, "best_val_loss_epoch": best_val_loss_epoch})

        # early stopping
        if early_stopper.early_stop(val_loss):             
            print("Early stopping at epoch {}".format(epoch+1))
            break

    # load best checkpoint
    if not config['is_debug']:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, f"{save_dir}/model_{best_val_loss_epoch}.pth", device)
        print(f"Loaded best checkpoint from epoch {best_val_loss_epoch} using VAL_LOSS metric")

        test(loader=test_loader, config=config, model=model, criterion=criterion, device=device)
    
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, f"{save_dir}/model_{best_val_dsc_epoch}.pth", device)
        print(f"Loaded best checkpoint from epoch {best_val_dsc_epoch} using VAL_DSC metric")

        test(loader=test_loader, config=config, model=model, criterion=criterion, device=device)



def train_epoch(loader, config, model, optimizer, criterion, device):
    model.train()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False,
                            return_with_label=config["dataset"]["instrument_labels"])
    post_transform = AsDiscrete(argmax=True, to_onehot=config['model']['n_classes'])

    losses = []
    
    for batch_idx, (images, masks, phases) in enumerate(tqdm(loader, total=len(loader))):        
        images = images.to(device)
        masks = masks.float().to(device)
        
        if config['condition']['phase'] == 'none':
            masks_pred = model(images)
        else:
            masks_pred = model(images, phases)

        loss = criterion(masks_pred, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"{batch_idx}/{len(loader)}, train_loss: {loss.item():.4f}")
        losses.append(loss.item())

        # masks = [mask for mask in decollate_batch(masks)]
        # masks_pred = torch.argmax(masks_pred, dim=1, keepdim=True)
        masks_pred = [post_transform(mask_pred) for mask_pred in decollate_batch(masks_pred)]
        masks_pred = torch.stack(masks_pred, dim=0)


        dice_metric(y_pred=masks_pred, y=masks)
        dice_metric_batch(y_pred=masks_pred, y=masks)
    
    
    dice_classwise_score = dice_metric_batch.aggregate()
    # dice_mean_score = dice_metric.aggregate().item()
    dice_mean_score = np.around(np.mean(list(dice_classwise_score.values())), decimals=4)
    
    loss = sum(losses) / len(losses)

    print(f"train_loss: {loss:.4f}, train_dice_score: {dice_mean_score:.4f}")

    for key in dice_classwise_score:
        print(f"{key}: {dice_classwise_score[key]}")

    print(f"The average DSC is: {np.mean(list(dice_classwise_score.values())):.4f}")

    dice_metric.reset()
    dice_metric_batch.reset()

    return loss, dice_mean_score, dice_classwise_score


def eval_epoch(loader, config, model, criterion, device):
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False,
                            return_with_label=config["dataset"]["instrument_labels"])
    post_transform = AsDiscrete(argmax=True, to_onehot=config['model']['n_classes'])

    losses = []

    with torch.no_grad():
        for batch_idx, (images, masks, phases) in enumerate(tqdm(loader, total=len(loader))):
            images = images.to(device)
            masks = masks.float().to(device)

            if config['condition']['phase'] == 'none':
                masks_pred = model(images)
            else:
                masks_pred = model(images, phases)

            loss = criterion(masks_pred, masks)

            losses.append(loss.item())

            # masks = [mask for mask in decollate_batch(masks)]
            # masks_pred = torch.argmax(masks_pred, dim=1, keepdim=True)
            masks_pred = [post_transform(mask_pred) for mask_pred in decollate_batch(masks_pred)]
            masks_pred = torch.stack(masks_pred, dim=0)
            
            dice_metric(y_pred=masks_pred, y=masks)
            dice_metric_batch(y_pred=masks_pred, y=masks)

    dice_classwise_score = dice_metric_batch.aggregate()
    # dice_mean_score = dice_metric.aggregate().item()
    dice_mean_score = np.around(np.mean(list(dice_classwise_score.values())), decimals=4)

    loss = sum(losses) / len(losses)

    print(f"val_loss: {loss:.4f}, val_dice_score: {dice_mean_score:.4f}")

    for key in dice_classwise_score:
        print(f"{key}: {dice_classwise_score[key]}")
    print(f"The average DSC is: {np.mean(list(dice_classwise_score.values())):.4f}")

    dice_metric.reset()
    dice_metric_batch.reset()

    return loss, dice_mean_score, dice_classwise_score