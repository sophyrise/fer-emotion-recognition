import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from dataset import make_loaders
from models import build


def check_output_shape(model, device, n_classes=7):
    model.eval()
    dummy = torch.zeros(2, 1, 48, 48).to(device)
    with torch.no_grad():
        out = model(dummy)
    expected = (2, n_classes)
    ok = out.shape == expected
    print(f"Output shape: {tuple(out.shape)}  {'OK' if ok else f'EXPECTED {expected}'}")
    return ok


def check_init_loss(model, loader, criterion, device):
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        out = model(x)
        loss = criterion(out, y)
    print(f"Initial loss: {loss.item():.4f}  (expected ~1.9459)")
    return loss.item()


def overfit_one_batch(model, loader, criterion, device, steps=200):
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        if (i + 1) % 50 == 0:
            acc = (out.argmax(1) == y).float().mean().item()
            print(f"  step {i+1:03d} | loss {loss.item():.4f} | acc {acc:.4f}")

    print("Overfit test done. Loss should be near 0, acc near 1.0")


def check_gradient_flow(model, loader, criterion, device):
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    print("\nGradient flow per layer:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name:40s} | grad mean: {param.grad.abs().mean().item():.6f}")
        else:
            print(f"  {name:40s} | NO GRADIENT")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv',  default='fer2013.csv')
    p.add_argument('--arch', default='plain',
                   choices=['tiny', 'plain', 'reg', 'resnet'])
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_ld, _, _ = make_loaders(args.csv, batch_size=32, num_workers=0)
    model = build(args.arch).to(device)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Check 0: Output Shape ---")
    check_output_shape(model, device)

    print("\n--- Check 1: Initial Loss ---")
    check_init_loss(model, train_ld, criterion, device)

    print("\n--- Check 2: Overfit One Batch ---")
    model2 = build(args.arch).to(device)
    overfit_one_batch(model2, train_ld, criterion, device)

    print("\n--- Check 3: Gradient Flow ---")
    model3 = build(args.arch).to(device)
    check_gradient_flow(model3, train_ld, criterion, device)
