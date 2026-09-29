import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

case_dir = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_193"
img_path = os.path.join(case_dir, "BraTS20_Training_193_flair.nii")
seg_path = os.path.join(case_dir, "BraTS20_Training_193_seg.nii")

img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))   # [z, y, x]
seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))   # labels: 0,1,2,4

z = img.shape[0] // 2
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img[z], cmap="gray")
plt.title(f"FLAIR slice z={z}")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img[z], cmap="gray")
plt.imshow(np.ma.masked_where(seg[z] == 0, seg[z]), cmap="jet", alpha=0.4)
plt.title("Overlay seg")
plt.axis("off")

plt.tight_layout()
plt.show()