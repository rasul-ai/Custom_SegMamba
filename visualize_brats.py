import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

case_id = "BraTS20_Training_193"
case_dir = f"data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/{case_id}"
out_path = f"{case_id}_modalities_overlay.png"

modalities = ["t1", "t1ce", "t2", "flair"]
titles = {"t1": "T1", "t1ce": "T1ce", "t2": "T2", "flair": "FLAIR"}

imgs = {m: sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_dir, f"{case_id}_{m}.nii")))
        for m in modalities}                                                   # [z, y, x]
seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_dir, f"{case_id}_seg.nii")))  # labels: 0,1,2,4

z = seg.shape[0] // 2

# fixed colour per BraTS label: 1 = necrotic/non-enhancing core, 2 = edema, 4 = enhancing tumor
labels = [1, 2, 4]
colors = ["#1f4fd8", "#2ec4d6", "#d62728"]
names = ["NCR/NET (1)", "Edema (2)", "Enhancing tumor (4)"]
cmap = ListedColormap(colors)
norm = BoundaryNorm([0.5, 1.5, 3, 4.5], cmap.N)

fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))

for col, m in enumerate(modalities):
    sl = imgs[m][z]
    vmax = np.percentile(sl[sl > 0], 99.5) if np.any(sl > 0) else sl.max()   # clip bright outliers per modality

    axes[0, col].imshow(sl, cmap="gray", vmin=0, vmax=vmax)
    axes[0, col].set_title(f"{titles[m]}  (z={z})")

    axes[1, col].imshow(sl, cmap="gray", vmin=0, vmax=vmax)
    axes[1, col].imshow(np.ma.masked_where(seg[z] == 0, seg[z]), cmap=cmap, norm=norm,
                        alpha=0.45, interpolation="nearest")
    axes[1, col].set_title(f"{titles[m]} + ground-truth labels")

for ax in axes.ravel():
    ax.axis("off")

fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors, names)],
           loc="lower center", ncol=3, frameon=False, fontsize=11)
fig.suptitle(case_id, fontsize=14)
plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.07, wspace=0.05, hspace=0.12)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
plt.show()
