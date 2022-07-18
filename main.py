import cv2, numpy as np
import SimpleITK as sitk
from skimage import measure

import nibabel as nib
import sys
import os

import time

time1 = time.time()

import datetime

# input_nii_path = r'D:\Code\textPY\skull.nii.gz'
# output_nii_path = r'/home/brainplan/YueNao/Data_text/CT/mid_dir/skull_output'+str(time1)+r'.nii'
input_nii_path = sys.argv[1]
output_nii_path = sys.argv[2]
mid_nii_path = sys.argv[3]


def window(img):
    nii_path = input_nii_path
    img0 = sitk.ReadImage(nii_path)
    Direction = img0.GetDirection()
    Origin = img0.GetOrigin()
    Spacing = img0.GetSpacing()

    for i in range(img.shape[0]):
        img[i] = 255.0 * (img[i] - 0) / (255 - 0)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255
        img[i] = img[i] - img[i].min()
        c = float(255) / img[i].max()
        img[i] = img[i] * c

    img_x = img.astype(np.uint8)
    img_x[img_x < 255] = 0
    img_x[img_x >= 255] = 1

    return img_x.astype(np.uint8)


def islands(image, sourImage):
    pred = nib.load(image)
    labels = pred.get_data()
    sourImage = nib.load(sourImage)

    labels = measure.label(labels, connectivity=2)

    max_num = 0
    print(np.max(labels) + 1)
    filemax = r'/home/brainplan/YueNao/Data_text/CT/OurData/time.txt'
    with open(filemax, 'a+') as fm:
        fm.write(str(np.max(labels) + 1) + '\n')
    for j in range(1, np.max(labels) + 1):
        print(str(j) + '/' + str(np.max(labels) + 1))
        if np.sum(labels == j) > max_num:
            max_num = np.sum(labels == j)
            max_pixel = j
        # print(np.sum(labels == j), np.sum(labels != 0))
        if np.sum(labels == j) > 0.1 * np.sum(labels != 0):
            labels[labels == j] = max_pixel

    labels[labels != max_pixel] = 0
    labels[labels == max_pixel] = 1

    nib.save(nib.Nifti1Image(labels.astype('uint8'), affine=sourImage.affine), output_nii_path)


if __name__ == "__main__":

    start = datetime.datetime.now()
    nii_path = input_nii_path
    img = sitk.ReadImage(nii_path)
    Direction = img.GetDirection()
    Origin = img.GetOrigin()
    Spacing = img.GetSpacing()

    img_x = sitk.GetArrayFromImage(img)
    img1_x = window(img_x)

    kernel = (3, 3)
    for i in range(img1_x.shape[0]):
        img1_x[i] = cv2.GaussianBlur(img1_x[i], kernel, 1)

    img1 = sitk.GetImageFromArray(img1_x)
    img1.SetDirection(Direction)
    img1.SetOrigin(Origin)
    img1.SetSpacing(Spacing)
    # sitk.WriteImage(img1, r'/home/brainplan/YueNao/Data_text/CT/mid_dir/skull_output_mid'+str(time1)+'.nii')
    sitk.WriteImage(img1, mid_nii_path + 'skull_mid.nii')

    im = mid_nii_path + 'skull_mid.nii'
    islands(im, input_nii_path)

    end = datetime.datetime.now()

    # filetime = r'/home/brainplan/YueNao/Data_text/CT/OurData/time.txt'
    # with open(filetime, 'a+') as ft:
    #    ft.write(str((end-start).seconds)+'\n')

    print("time is :" + str((end - start).seconds))
    print("finish")
