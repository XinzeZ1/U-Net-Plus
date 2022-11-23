import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
import time
import matplotlib.pyplot as plt

# count = 0
#
# for name in os.listdir('./Lung/Images_padded_test/'):
#
#     data = np.load('./Lung/Images_padded_test/' + name)
#     np.save('./data/test_data/lung_' + str(count) + '.npy', data)
#
#     mask = Image.open('./Lung/Mask_padded_test/' + name.split('.')[0] + '.png')
#     mask_ar = np.asarray(mask)
#     np.save('./data/test_data/mask_' + str(count) + '.npy', mask_ar)
#
#     count += 1
#     print('Have finished %d' % count)


import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
import SimpleITK as sitk
from nibabel.viewers import OrthoSlicer3D

file = './Data_Clean/BraTS19_CBICA_AAB_1/BraTS19_CBICA_AAB_1_t1_ventricles_Manual.nii.gz'  
# img = nib.load(file)

# print(img)
# print('**************************************')
# print(img.header)  # 
#
# width, height, queue = img.dataobj.shape
# print(img.dataobj.shape)
# print(img.dataobj[:, :, 0])
# print(img.dataobj.max)

image = sitk.ReadImage(file)
image_arr = sitk.GetArrayFromImage(image)
print(image_arr[55].max())
print(int(66.4))

# OrthoSlicer3D(img.dataobj).show()
#
# num = 1
# for i in range(0, queue, 10):
#     img_arr = img.dataobj[:, :, i]
#     plt.subplot(5, 4, num)
#     plt.imshow(img_arr, cmap='gray')
#     num += 1
#
# plt.show()
