"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import argparse
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils

def get_args():
    parser = argparse.ArgumentParser(description="Train TinyVGG model")

    parser.add_argument("--epochs",       type=int,   default=5,     help="Number of epochs")
    parser.add_argument("--batch_size",   type=int,   default=32,    help="Batch size")
    parser.add_argument("--hidden_units", type=int,   default=10,    help="Hidden units in TinyVGG")
    parser.add_argument("--lr",           type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train_dir",    type=str,   default="data/pizza_steak_sushi/train")
    parser.add_argument("--test_dir",     type=str,   default="data/pizza_steak_sushi/test")

    return parser.parse_args()

def main():
    args = get_args()

    # Print what we're using
    print(f"[INFO] Epochs:       {args.epochs}")
    print(f"[INFO] Batch size:   {args.batch_size}")
    print(f"[INFO] Hidden units: {args.hidden_units}")
    print(f"[INFO] LR:           {args.lr}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        transform=data_transform,
        batch_size=args.batch_size       # 👈 from args
    )

    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=args.hidden_units,  # 👈 from args
        output_shape=len(class_names)
    ).to(device)

    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 👈 from args

    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,              # 👈 from args
        device=device
    )

    utils.save_model(
        model=model,
        target_dir="models",
        model_name="05_going_modular_script_mode_tinyvgg_model.pth"
    )

if __name__ == "__main__":
    main()