import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MEAN, STD = [0.5077], [0.2550]


class FER2013(Dataset):
    def __init__(self, df, transform=None):
        self.pixels = df['pixels'].values
        self.labels = df['emotion'].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        arr = np.array(self.pixels[i].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def get_transforms(augment=False):
    base = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if augment:
        aug = [
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.RandomCrop(48, padding=4),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return T.Compose(aug + base)
    return T.Compose(base)


def make_loaders(csv_path, batch_size=64, augment=False, num_workers=2):
    import pandas as pd
    from torch.utils.data import DataLoader
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    tr = df[df['Usage'] == 'Training']
    va = df[df['Usage'] == 'PublicTest']
    te = df[df['Usage'] == 'PrivateTest']
    train_ds = FER2013(tr, get_transforms(augment))
    val_ds   = FER2013(va, get_transforms(False))
    test_ds  = FER2013(te, get_transforms(False))
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size, shuffle=False, **kw),
    )
