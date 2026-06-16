"""Generate per-architecture result charts for the README from logged WandB metrics.

All numbers are taken from the final / best-checkpoint values reported in the
fer2013 WandB project (see README tables). Run:  python notebooks/make_readme_charts.py
"""
import os
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "data")


def label_bars(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9,
                    xytext=(0, 2), textcoords="offset points")


# ---- Arch 1: TinyMLP - per-class F1 (underfitting, weak across the board) ----
classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
tiny_f1 = [0.299, 0.333, 0.283, 0.601, 0.347, 0.613, 0.392]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(classes, tiny_f1, color="#9467bd")
label_bars(ax, bars)
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1")
ax.set_title("TinyMLP - Per-Class F1 (macro-F1 = 0.420)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tiny.png"), dpi=120)
plt.close(fig)

# ---- Arch 2: PlainCNN - train vs val (accuracy + loss), the overfitting gap ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
b1 = axes[0].bar(["train", "val"], [0.984, 0.575], color=["#1f77b4", "#ff7f0e"])
label_bars(axes[0], b1)
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("PlainCNN - Accuracy (train vs val)")
b2 = axes[1].bar(["train", "val"], [0.081, 2.196], color=["#1f77b4", "#ff7f0e"])
label_bars(axes[1], b2)
axes[1].set_ylim(0, 2.5)
axes[1].set_ylabel("Loss")
axes[1].set_title("PlainCNN - Loss (train vs val)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plain.png"), dpi=120)
plt.close(fig)

# ---- Arch 3: RegCNN - hyperparameter search (test acc + macro-F1 per run) ----
reg_runs = ["reg_baseline", "reg_lr1e-4", "reg_dropout06",
            "reg_sgd_cosine", "reg_no_aug", "reg_weighted"]
reg_acc = [0.669, 0.683, 0.683, 0.659, 0.640, 0.655]
reg_f1 = [0.633, 0.652, 0.640, 0.547, 0.617, 0.629]
x = range(len(reg_runs))
w = 0.4
fig, ax = plt.subplots(figsize=(10, 4.5))
b1 = ax.bar([i - w / 2 for i in x], reg_acc, w, label="Test Acc", color="#1f77b4")
b2 = ax.bar([i + w / 2 for i in x], reg_f1, w, label="Test Macro-F1", color="#ff7f0e")
label_bars(ax, b1)
label_bars(ax, b2)
ax.set_ylim(0, 0.8)
ax.set_xticks(list(x))
ax.set_xticklabels(reg_runs, rotation=20, ha="right")
ax.set_title("RegCNN - Hyperparameter Search")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reg.png"), dpi=120)
plt.close(fig)

# ---- Arch 4: MiniResNet - hyperparameter search (test acc + macro-F1 per run) ----
res_runs = ["resnet_baseline", "resnet_no_smooth", "resnet_bs32", "resnet_plateau"]
res_acc = [0.690, 0.689, 0.688, 0.674]
res_f1 = [0.665, 0.672, 0.669, 0.645]
x = range(len(res_runs))
fig, ax = plt.subplots(figsize=(9, 4.5))
b1 = ax.bar([i - w / 2 for i in x], res_acc, w, label="Test Acc", color="#1f77b4")
b2 = ax.bar([i + w / 2 for i in x], res_f1, w, label="Test Macro-F1", color="#ff7f0e")
label_bars(ax, b1)
label_bars(ax, b2)
ax.set_ylim(0, 0.8)
ax.set_xticks(list(x))
ax.set_xticklabels(res_runs, rotation=20, ha="right")
ax.set_title("MiniResNet - Hyperparameter Search")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "resnet.png"), dpi=120)
plt.close(fig)

print("wrote tiny.png, plain.png, reg.png, resnet.png to data/")