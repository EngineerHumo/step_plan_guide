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


def default_device() -> str:
    """Prefer CUDA when available, defaulting to GPU 0 for multi-GPU training."""

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tensor_to_image(tensor: torch.Tensor) -> "np.ndarray":  # type: ignore[name-defined]
    import numpy as np

    array = tensor.detach().cpu().clamp(0, 1)
    array = array.permute(1, 2, 0).numpy()
    return (array * 255).astype(np.uint8)


def log_to_visdom(
    viz: Optional["visdom.Visdom"],
    epoch: int,
    phase: str,
    batch_images: torch.Tensor,
    batch_heatmaps: torch.Tensor,
    batch_masks: torch.Tensor,
    batch_preds: torch.Tensor,
) -> None:
    """Visualize a random sample from the batch on Visdom."""

    if viz is None or batch_images.shape[0] == 0:
        return

    def _prep_single(t: torch.Tensor) -> torch.Tensor:
        tensor = t.detach().cpu().clamp(0, 1)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    idx = int(torch.randint(0, batch_images.shape[0], (1,)).item())
    img = _prep_single(batch_images[idx])
    heatmap = _prep_single(batch_heatmaps[idx])
    gt = _prep_single(batch_masks[idx])
    pred = _prep_single(batch_preds[idx])

    viz.image(img, win="input_image", opts={"title": f"{phase} Epoch {epoch} - Input"})
    viz.image(heatmap, win="heatmap", opts={"title": f"{phase} Epoch {epoch} - Heatmap"})
    viz.image(gt, win="ground_truth", opts={"title": f"{phase} Epoch {epoch} - Ground Truth"})
    viz.image(pred, win="prediction", opts={"title": f"{phase} Epoch {epoch} - Prediction"})


def save_validation_batch(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
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

        image_with_click = image_np.copy()
        if heatmap_np.max() > 0:
            click_y, click_x = divmod(heatmap_np.argmax(), heatmap_np.shape[1])
            cv2.circle(image_with_click, (int(click_x), int(click_y)), 8, (255, 0, 0), thickness=-1)

        pred_mask = (pred_np > 0.5).astype(np.uint8) * 255
        gt_mask = (mask_np > 0.5).astype(np.uint8) * 255

        basename = f"sample_{batch_idx:03d}_{i:02d}"
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_image.png"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_click.png"), cv2.cvtColor(image_with_click, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_pred.png"), pred_mask)
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_gt.png"), gt_mask)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dice = dice_coefficient(prob, target).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    return (1 - dice) + 0.5 * bce


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_root: Optional[str] = None,
    epoch: Optional[int] = None,
    viz: Optional["visdom.Visdom"] = None,
    phase: str = "Val",
) -> tuple[float, float]:
    model.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for batch_idx, (images, heatmaps, masks) in enumerate(loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            masks = masks.to(device)
            logits = model(images, heatmaps)
            prob = torch.sigmoid(logits)
            dice_scores.append(dice_coefficient(prob, masks).mean().item())
            iou_scores.append(iou_score(prob, masks).mean().item())

            if viz is not None and epoch is not None:
                log_to_visdom(viz, epoch, phase, images, heatmaps, masks, prob)

            if save_root and epoch is not None:
                save_validation_batch(
                    images=images,
                    heatmaps=heatmaps,
                    masks=masks,
                    preds=prob,
                    save_root=save_root,
                    epoch=epoch,
                    batch_idx=batch_idx,
                )
    model.train()
    return float(sum(dice_scores) / len(dice_scores)), float(sum(iou_scores) / len(iou_scores))


def train(
    train_dir: str,
    val_dir: Optional[str],
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 5e-4,
    num_workers: int = 4,
    device: str = default_device(),
    use_visdom: bool = False,
    visdom_env: str = "prp_segmentation",
    visdom_port: int = 8097,
    output_dir: str = "output",
):
    device = torch.device(device)
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, "train_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")

    def log_message(message: str) -> None:
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    train_dataset = PRPDataset(train_dir, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_dataset = PRPDataset(val_dir, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = PRPSegmenter()
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        log_message("Using DataParallel on GPUs: 0 and 1")
        model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    viz = None
    if use_visdom:
        import visdom

        viz = visdom.Visdom(env=visdom_env, port=visdom_port)
        if not viz.check_connection():
            log_message("[Visdom] Connection failed. Visualizations will be skipped.")
            viz = None

    best_val_dice = float("-inf")
    best_models: list[tuple[float, int, str]] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, heatmaps, masks in progress:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            masks = masks.to(device)

            logits = model(images, heatmaps)
            loss = dice_bce_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

            prob = torch.sigmoid(logits)
            log_to_visdom(viz, epoch, "Train", images, heatmaps, masks, prob)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        log_message(f"Epoch {epoch}: Train Loss={avg_loss:.4f}")

        if val_loader:
            val_save_dir = os.path.join(output_dir, "val_outputs")
            val_dice, val_iou = evaluate(
                model, val_loader, device, save_root=val_save_dir, epoch=epoch, viz=viz, phase="Val"
            )
            log_message(f"Epoch {epoch}: Val Dice={val_dice:.4f} | Val IoU={val_iou:.4f}")

            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            candidate_name = f"best_model_epoch{epoch:03d}_dice{val_dice:.4f}.pth"
            candidate_path = os.path.join(output_dir, candidate_name)
            torch.save(model_to_save.state_dict(), candidate_path)
            best_models.append((val_dice, epoch, candidate_path))
            best_models.sort(key=lambda item: item[0], reverse=True)
            if len(best_models) > 3:
                _, _, drop_path = best_models.pop()
                if os.path.exists(drop_path):
                    log_message(f"Removed checkpoint outside top-3: {os.path.basename(drop_path)}")
                    os.remove(drop_path)
            log_message(f"Checkpoint considered for top-3: {candidate_name}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, "best_model.pth"))
                log_message(f"New best model saved with Val Dice {val_dice:.4f}")

        train_dice, train_iou = evaluate(model, train_loader, device)
        log_message(f"Epoch {epoch}: Train Dice={train_dice:.4f} | Train IoU={train_iou:.4f}")

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, "final_model.pth"))
    log_message("Training completed. Final model saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive PRP area segmentation trainer")
    parser.add_argument("--train_dir", type=str, default="dataset/train", help="Path to training dataset")
    parser.add_argument("--val_dir", type=str, default="dataset/val", help="Path to validation dataset")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=default_device())
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
