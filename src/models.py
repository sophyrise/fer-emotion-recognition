import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, n=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, 128), nn.ReLU(),
            nn.Linear(128, n)
        )

    def forward(self, x):
        return self.net(x)


class PlainCNN(nn.Module):
    def __init__(self, n=7):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
            )

        self.feat = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128))
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(),
            nn.Linear(256, n)
        )

    def forward(self, x):
        return self.clf(self.feat(x))


class RegCNN(nn.Module):
    def __init__(self, n=7, p=0.5):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.feat = nn.Sequential(blk(1, 64), blk(64, 128), blk(128, 256))
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p), nn.Linear(256 * 6 * 6, 512), nn.ReLU(),
            nn.Dropout(p), nn.Linear(512, n)
        )

    def forward(self, x):
        return self.clf(self.feat(x))


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class MiniResNet(nn.Module):
    def __init__(self, n=7, p=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.stage1 = self._stage(64, 128)
        self.stage2 = self._stage(128, 256)
        self.stage3 = self._stage(256, 512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(512, n)
        )

    @staticmethod
    def _stage(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            _ResBlock(out_ch)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.clf(self.gap(x))


def build(name, **kw):
    n = kw.get('n', 7)
    p = kw.get('p', 0.5)
    if name == 'tiny':
        return TinyMLP(n=n)
    elif name == 'plain':
        return PlainCNN(n=n)
    elif name == 'reg':
        return RegCNN(n=n, p=p)
    elif name == 'resnet':
        return MiniResNet(n=n, p=kw.get('p', 0.4))
    else:
        raise ValueError(f"Unknown arch: {name}")
