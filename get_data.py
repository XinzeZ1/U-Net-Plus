import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import cv2


def split_data(input_path, train_data_save_path, test_data_save_path):
    slice_count = 0

    num = 0
    count = 0
    len_name = []
    for name in os.listdir(input_path):
        len_name.append(name)
        if len(len_name) < 70:
            number_name = len(len_name)
            image_path = input_path+name+'/BraTS19_CBICA_' + name.split('_')[2] +'_1_t1.nii.gz'
            mask_path = input_path+name+'/BraTS19_CBICA_' + name.split('_')[2] +'_1_t1_ventricles_Manual.nii.gz'
            image = sitk.ReadImage(image_path)
            image_arr = sitk.GetArrayFromImage(image)

            mask = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask)

            slice_num, high, width = np.shape(image_arr)
            slice_count += slice_num

            for j in range(slice_num):

                if np.sum(mask_arr[j]) > 0:

                    save_image = np.reshape(image_arr[j], (240, 240, 1))
                    save_mask = np.reshape(mask_arr[j], (240, 240, 1))
                    save_image[save_image < -200] = -200
                    save_image[save_image > 255] = 255

                    np.save(train_data_save_path+'Brain_' + str(num) + '.npy', save_image)
                    np.save(train_data_save_path+'/mask_' + str(num) + '.npy', save_mask)
                    print('series {} slice:{} saved'''.format(number_name, num))
                    num += 1
                    count += 1
                else:
                    count += 1

                print('until volume {} have {} examples'''.format(number_name, count))
            print('total num of examples {}'''.format(count))
            print('total num of slices {}'''.format(slice_count))

            num = 0
        else:
            image_path = input_path + name + '/BraTS19_CBICA_' + name.split('_')[2] +'_1_t1.nii.gz'
            mask_path = input_path + name + '/BraTS19_CBICA_' + name.split('_')[2] +'_1_t1_ventricles_Manual.nii.gz'

            image = sitk.ReadImage(image_path)
            image_arr = sitk.GetArrayFromImage(image)

            mask = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask)

            slice_num, high, width = np.shape(image_arr)
            slice_count += slice_num

            for j in range(slice_num):

                if np.sum(mask_arr[j]) > 0:

                    save_image = np.reshape(image_arr[j], (240, 240, 1))

                    save_mask = np.reshape(mask_arr[j], (240, 240, 1))
                    save_image[save_image < -200] = -200
                    save_image[save_image > 255] = 255

                    np.save(test_data_save_path+'/Brain_' + str(num), save_image)
                    np.save(test_data_save_path+'/mask_' + str(num), save_mask)
                    print('series {} slice:{} saved'''.format(len(len_name), num))
                    num += 1
                    count += 1
                else:
                    count += 1
            print('until volume {} have {} examples'''.format(len(len_name), count))
        print('total num of examples {}'''.format(count))
        print('total num of slices {}'''.format(slice_count))


if __name__ == '__main__':
    input_path = './Data_Clean/'
    train_data_save_path = './data/train_data/'
    test_data_save_path = './data/test_data/'

    split_data(input_path, train_data_save_path, test_data_save_path)
