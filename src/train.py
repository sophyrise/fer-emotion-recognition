import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import wandb
import torch.nn as nn
from sklearn.metrics import f1_score

from dataset import make_loaders, EMOTIONS, MEAN, STD
from models import build
from engine import train_one_epoch, evaluate


def _parse_bool(v):
    return v.lower() in ('1', 'true', 'yes')


def log_prediction_table(model, loader, device, n=32):
    model.eval()
    imgs, preds, gts = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            imgs.extend(x.cpu())
            preds.extend(out.argmax(1).cpu().tolist())
            gts.extend(y.tolist())
            if len(imgs) >= n:
                break

    table = wandb.Table(columns=['image', 'ground_truth', 'prediction', 'correct'])
    for img, gt, pred in zip(imgs[:n], gts[:n], preds[:n]):
        arr = img.squeeze().numpy()
        arr = np.clip((arr * STD[0] + MEAN[0]) * 255, 0, 255).astype(np.uint8)
        table.add_data(
            wandb.Image(arr),
            EMOTIONS[gt],
            EMOTIONS[pred],
            gt == pred,
        )
    wandb.log({'val_predictions': table})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',          default='fer2013.csv')
    p.add_argument('--arch',         default='reg',
                   choices=['tiny', 'plain', 'reg', 'resnet'])
    p.add_argument('--group',        default='baseline')
    p.add_argument('--name',         default='run1')
    p.add_argument('--epochs',       type=int,         default=40)
    p.add_argument('--lr',           type=float,       default=1e-3)
    p.add_argument('--batch',        type=int,         default=64)
    p.add_argument('--wd',           type=float,       default=0.0)
    p.add_argument('--dropout',      type=float,       default=0.5)
    p.add_argument('--label_smooth', type=float,       default=0.0)
    p.add_argument('--augment',      type=_parse_bool, default=False)
    p.add_argument('--optimizer',    default='adam',   choices=['adam', 'sgd'])
    p.add_argument('--sched',        default='plateau', choices=['plateau', 'cosine'])
    p.add_argument('--workers',      type=int,         default=2)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    wandb.init(
        project='fer2013',
        group=args.group,
        name=args.name,
        tags=[args.arch],
        config=vars(args),
    )

    train_ld, val_ld, test_ld = make_loaders(
        args.csv, args.batch, args.augment, args.workers
    )
    model = build(args.arch, p=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.wd)

    if args.sched == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3)

    best_val_acc = 0.0

    for ep in range(args.epochs):
        tl, ta, grad_norm = train_one_epoch(model, train_ld, criterion, opt, device)
        vl, va, gts, preds = evaluate(model, val_ld, criterion, device)

        if args.sched == 'cosine':
            sched.step()
        else:
            sched.step(vl)

        f1_macro   = f1_score(gts, preds, average='macro', zero_division=0)
        f1_per_cls = f1_score(gts, preds, average=None,    zero_division=0)

        log = {
            'epoch':      ep,
            'train_loss': tl,
            'train_acc':  ta,
            'val_loss':   vl,
            'val_acc':    va,
            'f1_macro':   f1_macro,
            'grad_norm':  grad_norm,
            'lr':         opt.param_groups[0]['lr'],
        }
        for i, cls in enumerate(EMOTIONS):
            log[f'f1_{cls}'] = f1_per_cls[i]

        wandb.log(log)
        print(f"Ep {ep:02d} | train {tl:.4f}/{ta:.4f} | "
              f"val {vl:.4f}/{va:.4f} | f1 {f1_macro:.4f} | gnorm {grad_norm:.2f}")

        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), 'best.pth')
            print(f"  -> Saved best (val_acc={va:.4f})")

    model.load_state_dict(torch.load('best.pth', map_location=device))
    _, test_acc, gts, preds = evaluate(model, test_ld, criterion, device)
    f1_test = f1_score(gts, preds, average='macro', zero_division=0)
    print(f"Test acc: {test_acc:.4f} | Test F1: {f1_test:.4f}")

    log_prediction_table(model, val_ld, device)

    wandb.log({
        'test_acc': test_acc,
        'test_f1':  f1_test,
        'conf_mat': wandb.plot.confusion_matrix(
            y_true=gts, preds=preds, class_names=EMOTIONS
        ),
    })

    artifact = wandb.Artifact(f'{args.arch}-{args.name}', type='model')
    artifact.add_file('best.pth')
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == '__main__':
    main()
