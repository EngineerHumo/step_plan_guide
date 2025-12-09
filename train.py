import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PRPDataset
from model import PRPSegmenter
from utils import dice_coefficient, iou_score


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tensor_to_image(tensor: torch.Tensor) -> "np.ndarray":  # type: ignore[name-defined]
    import numpy as np

    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255).astype(np.uint8)


def save_validation_batch(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    aux_masks: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    save_root: str,
    epoch: int,
    batch_idx: int,
) -> None:
    import cv2
    import numpy as np

    epoch_dir = os.path.join(save_root, f"epoch_{epoch:03d}")
    ensure_dir(epoch_dir)

    for i in range(images.shape[0]):
        image_np = tensor_to_image(images[i])
        heatmap_np = heatmaps[i, 0].detach().cpu().numpy()
        mask_np = masks[i, 0].detach().cpu().numpy()
        pred_np = preds[i, 0].detach().cpu().numpy()

        click_y, click_x = divmod(heatmap_np.argmax(), heatmap_np.shape[1])
        image_with_click = image_np.copy()
        cv2.circle(image_with_click, (int(click_x), int(click_y)), 8, (255, 0, 0), thickness=-1)

        pred_mask = (pred_np > 0.5).astype(np.uint8) * 255
        gt_mask = (mask_np > 0.5).astype(np.uint8) * 255

        basename = f"sample_{batch_idx:03d}_{i:02d}"
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_image.png"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_click.png"), cv2.cvtColor(image_with_click, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_pred.png"), pred_mask)
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_gt.png"), gt_mask)


def dice_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(min=1e-6, max=1 - 1e-6)
    dice = dice_coefficient(pred, target).mean()
    bce = nn.functional.binary_cross_entropy(pred, target)
    return (1 - dice) + 0.5 * bce


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_root: Optional[str] = None,
    epoch: Optional[int] = None,
) -> tuple[float, float]:
    model.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for batch_idx, (images, heatmaps, aux_masks, masks) in enumerate(loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            aux_masks = aux_masks.to(device)
            masks = masks.to(device)
            preds = model(images, heatmaps, aux_masks)
            dice_scores.append(dice_coefficient(preds, masks).mean().item())
            iou_scores.append(iou_score(preds, masks).mean().item())

            if save_root and epoch is not None:
                save_validation_batch(
                    images=images,
                    heatmaps=heatmaps,
                    aux_masks=aux_masks,
                    masks=masks,
                    preds=preds,
                    save_root=save_root,
                    epoch=epoch,
                    batch_idx=batch_idx,
                )
    model.train()
    return float(sum(dice_scores) / len(dice_scores)), float(sum(iou_scores) / len(iou_scores))


def train(
    train_dir: str,
    val_dir: Optional[str],
    epochs: int = 50,
    batch_size: int = 2,
    lr: float = 1e-4,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_visdom: bool = False,
    visdom_env: str = "prp_segmentation",
    visdom_port: int = 8097,
    output_dir: str = "output",
):
    device = torch.device(device)
    train_dataset = PRPDataset(train_dir, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_dataset = PRPDataset(val_dir, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = PRPSegmenter().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    viz = None
    if use_visdom:
        import visdom

        viz = visdom.Visdom(env=visdom_env, port=visdom_port)
        if not viz.check_connection():
            print("[Visdom] Connection failed. Visualizations will be skipped.")
            viz = None

    ensure_dir(output_dir)
    best_val_dice = float("-inf")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, heatmaps, aux_masks, masks in progress:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            aux_masks = aux_masks.to(device)
            masks = masks.to(device)

            preds = model(images, heatmaps, aux_masks)
            loss = dice_bce_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

            if viz is not None:
                viz.image(images[0].cpu(), win="input_image", opts={"title": f"Input Epoch {epoch}"})
                viz.image(heatmaps[0].cpu(), win="heatmap", opts={"title": f"Heatmap Epoch {epoch}"})
                viz.image(aux_masks[0].cpu(), win="auxiliary", opts={"title": f"Auxiliary Epoch {epoch}"})
                viz.image(masks[0].cpu(), win="ground_truth", opts={"title": f"Mask Epoch {epoch}"})
                viz.image(preds[0].detach().cpu(), win="prediction", opts={"title": f"Prediction Epoch {epoch}"})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}")

        if val_loader:
            val_save_dir = os.path.join(output_dir, "val_outputs")
            val_dice, val_iou = evaluate(model, val_loader, device, save_root=val_save_dir, epoch=epoch)
            print(f"Epoch {epoch}: Val Dice={val_dice:.4f} | Val IoU={val_iou:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                print(f"New best model saved with Val Dice {val_dice:.4f}")

        train_dice, train_iou = evaluate(model, train_loader, device)
        print(f"Epoch {epoch}: Train Dice={train_dice:.4f} | Train IoU={train_iou:.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive PRP area segmentation trainer")
    parser.add_argument("--train_dir", type=str, default="dataset/train", help="Path to training dataset")
    parser.add_argument("--val_dir", type=str, default="dataset/val", help="Path to validation dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_visdom", action="store_true", help="Enable Visdom visualization")
    parser.add_argument("--visdom_env", type=str, default="prp_segmentation", help="Visdom environment name")
    parser.add_argument("--visdom_port", type=int, default=8097, help="Visdom server port")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save validation outputs and models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        use_visdom=args.use_visdom,
        visdom_env=args.visdom_env,
        visdom_port=args.visdom_port,
        output_dir=args.output_dir,
    )
