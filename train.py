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
    """Custom collate function to handle variable-length point annotations."""
    imgs, densities, points = zip(*batch)
    imgs = torch.stack(imgs, 0)
    densities = torch.stack(densities, 0)
    return imgs, densities, list(points)


def train(dataset_name, batch_size, epochs, lr, save_dir, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_data = load_dataset(name=dataset_name, split="train")
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )

    print(f"✅ {len(train_data)} image-GT pairs loaded from: {train_data.root}")

    # Initialize model
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

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, gt_densities, _ in loop:
            imgs = imgs.to(device)
            gt_densities = gt_densities.to(device)

            # Forward pass
            final_output, reg_output, det_output, att_map = model(imgs)

            # Loss computation
            loss = criterion(final_output, gt_densities)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")

    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        resume=args.resume
    )
