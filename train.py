import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.decidenet_model import DecideNet
from my_dataset import load_dataset


def custom_collate(batch):
    imgs, densities, points = zip(*batch)
    imgs = torch.stack(imgs, 0)
    densities = torch.stack(densities, 0)
    return imgs, densities, list(points)


def train(dataset_name, batch_size, epochs, lr, save_dir, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_dataset(name=dataset_name, split="train")
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )

    print(f"✅ {len(train_data)} image-GT pairs loaded from: {train_data.root}")

    model = DecideNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    start_epoch = 0
    if resume and os.path.exists(resume):
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔁 Resumed training from checkpoint at epoch {start_epoch}")

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for imgs, gt_densities, points_list in loop:
            imgs = imgs.to(device)
            gt_densities = gt_densities.to(device)

            # Create detection targets
            targets = []
            for points in points_list:
                boxes = []
                labels = []
                for pt in points:
                    x, y = pt
                    boxes.append([x, y, x + 1, y + 1])  # small box around each point
                    labels.append(1)  # 1 = person
                targets.append({
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(labels, dtype=torch.int64).to(device)
                })

            optimizer.zero_grad()

            # Forward pass
            final_output, reg_output, det_output, att_map, det_losses = model(imgs, targets)

            # Combine losses: regression + detection + (optional attention)
            reg_loss = criterion(final_output, gt_densities)
            det_loss = sum(loss.detach() for loss in det_losses.values())  # detach detection loss
            total_loss = reg_loss + det_loss  # optionally add attention loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")

        # Save model
        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, save_path)
        print(f"💾 Checkpoint saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DecideNet for crowd counting")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: shanghaitech")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        resume=args.resume
    )
