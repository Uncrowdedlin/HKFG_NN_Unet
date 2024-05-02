import nibabel as nib
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import os
import pandas as pd

def calculate_metrics(inferred_file, expert_file):
    inferred_img = nib.load(inferred_file)
    expert_img = nib.load(expert_file)

    inferred_data = inferred_img.get_fdata()
    expert_data = expert_img.get_fdata()

    inferred_data[inferred_data > 0] = 1
    expert_data[expert_data > 0] = 1

    # 计算Dice系数
    intersection = np.sum(inferred_data * expert_data)
    union = np.sum(inferred_data) + np.sum(expert_data)
    dice = 2 * intersection / union

    # 计算PPV（Positive Predictive Value）
    ppv = np.sum(inferred_data * expert_data) / np.sum(inferred_data)

    # 计算Jaccard系数
    jaccard = intersection / (np.sum(inferred_data) + np.sum(expert_data) - intersection)
    return dice, ppv, jaccard


def calculate_hd95(inferred_file, expert_file):
    inferred_img = nib.load(inferred_file)
    expert_img = nib.load(expert_file)

    inferred_data = inferred_img.get_fdata()
    expert_data = expert_img.get_fdata()

    inferred_data[inferred_data > 0] = 1
    expert_data[expert_data > 0] = 1

    hd_distances = []
    for i in range(inferred_data.shape[2]):
        inferred_slice = inferred_data[:, :, i]
        expert_slice = expert_data[:, :, i]

        inferred_indices = np.argwhere(inferred_slice == 1)
        expert_indices = np.argwhere(expert_slice == 1)

        hd_distance = max(directed_hausdorff(inferred_indices, expert_indices)[0],
                          directed_hausdorff(expert_indices, inferred_indices)[0])
        hd_distances.append(hd_distance)

    hd95 = np.percentile(hd_distances, 95)

    return hd95


for i in range(1,11):
 # 输入推理出来的.nii图像和专家手动分割出的.nii图像的文件路径
 inferred_file = f"/media/hyhz/软件/HJR2024/nnUNet-2.4.2/nnUNet/predict/s00{i}.nii.gz"
 expert_file = f"/media/hyhz/软件/HJR2024/nnUNet-2.4.2/1111/s00{i}_mask.nii.gz"

 dice, ppv, jaccard = calculate_metrics(inferred_file, expert_file)
 hd95 = calculate_hd95(inferred_file, expert_file)



 print(f"第{i}个样本的评估指标",end='     ')
 print("Dice:", dice,end='     ')
 print("PPV:", ppv,end='     ')
 print("Jaccard:", jaccard,end='     ')
 print("Hd95:", hd95,end='     ')
 print('\n')
