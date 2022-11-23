import numpy as np
import os
from PIL import Image
from skimage.transform import rescale, resize
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transform(image, alpha, sigma, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    # dz = np.zeros_like(dx)

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    # print(x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


def elastic(image, alpha, sigma):
    random_stae = np.random.RandomState(42)
    size = np.shape(image)
    elastic_image = np.zeros(size)
    for i in range(size[2]):
        slice_image = image[:, :, i]
        image2 = elastic_transform(slice_image, alpha, sigma, random_stae)
        elastic_image[:, :, i] = image2
    return elastic_image


def zoom(image, factor, fill_color):
    size = np.shape(image)
    zoomed_image = np.zeros(size)
    for i in range(size[2]):
        slice_image = image[:, :, i]
        slice_image = Image.fromarray(slice_image)
        new_hight = size[0]*factor
        new_width = size[1]*factor

        center = [size[0]/2, size[1]/2]
        x1 = int(center[0] - new_width/2)
        y1 = int(center[1] - new_hight/2)
        x2 = int(center[0] + new_width/2)
        y2 = int(center[1] + new_hight/2)
        image2 = np.array(slice_image.transform((size[0], size[1]), Image.EXTENT, [x1, y1, x2, y2], fillcolor=fill_color))
        zoomed_image[:, :, i] = image2
    return zoomed_image


def rotate_image(image, fill_color):
    num = np.random.randint(0, 15, 1)
    size = np.shape(image)
    rotated_image = np.zeros(size)
    for i in range(size[2]):
        slice_image = image[:, :, i]
        slice_image = Image.fromarray(slice_image)
        rotate_image = slice_image.rotate(num, fillcolor=fill_color)
        rotated_image[:, :, i] = rotate_image
    return rotated_image


def flip(image, axis):
    size = np.shape(image)
    zoomed_image = np.zeros(size)
    for i in range(size[2]):
        if axis == 0:
            image2 = np.fliplr(image[:, :, i])
            zoomed_image[:, :, i] = image2
        elif axis == 1:
            image2 = np.flipud(image[:, :, i])
            zoomed_image[:, :, i] = image2
    return zoomed_image


def read_train_data(i, path, batch_size, random_array, order):

    dataset_num = int(len([name for name in os.listdir(path)]) / 2)
    # start = time.time()
    if order == 'ordered':
        feed_volume = []
        feed_mask = []
        for j in range(batch_size):
            # ramdom_num = np.random.uniform(0.0, 1, 1)
            # ramdom_num1 = np.random.uniform(0.0, 1, 1)
            # ramdom_num2 = np.random.uniform(0.0, 1, 1)
            # ramdom_num3 = np.random.uniform(0.0, 1, 1)

            volume = np.load(path + 'Brain_' + str((i * batch_size + j) % dataset_num) + '.npy').reshape((240, 240, 1))
            # volume = np.resize(volume, [256, 256, 3])
            mask = np.load(path + 'mask_' + str((i * batch_size + j) % dataset_num) + '.npy').reshape((240, 240, 1))
            # mask = np.resize(mask, [256, 256, 3])

            feed_volume.append(volume)
            feed_mask.append(mask)

        return np.stack(feed_volume), np.stack(feed_mask)


def read_data(i, data_type, input_path):

    if data_type == 'Brain':
        volume = np.load(input_path + 'Brain_' + str(i) + '.npy')

        data_3 = volume.reshape(1, 240, 240, 1)

    elif data_type == 'mask':
        mask = np.load(input_path + 'mask_' + str(i) + '.npy')
        data_3 = mask.reshape(1, 240, 240, 1)
    return data_3


# def read_data_normal_size(i, data_type, input_path):
#     if data_type == 'volume':
#         volume = np.load(input_path + 'volume_' + str(i) + '.npz')
#         data = volume.f.arr_0
#
#         data_1 = data.reshape(1, 512, 512, 3)
#
#     elif data_type == 'mask':
#         mask = np.load(input_path + 'mask_' + str(i) + '.npz')
#         data = mask.f.arr_0
#         data_1 = data.reshape(1, 512, 512, 3)
#     return data_1


def ofline_augument(data_path, save_path):
    data_size = int(len([name for name in os.listdir(data_path)]) / 2)
    k = 0
    for j in range(data_size):
        # ramdom_num = np.random.uniform(0.0, 1, 1)
        # ramdom_num1 = np.random.uniform(0.0, 1, 1)
        # ramdom_num2 = np.random.uniform(0.0, 1, 1)
        # ramdom_num3 = np.random.uniform(0.0, 1, 1)

        volume = np.load(data_path + 'volume_' + str(j) + '.npz')
        volume = volume.f.arr_0
        mask = np.load(data_path + 'mask_' + str(j) + '.npz')
        mask = mask.f.arr_0

        # if ramdom_num > 0 and ramdom_num < 0.5:
        #     volume = flip(volume, 0)
        #     mask = flip(mask, 0)
        #     k += 1
        volume = elastic(volume, 2000, 40)
        mask = elastic(mask, 2000, 40)
        np.savez_compressed(save_path + '/volume_' + str(k), volume)
        np.savez_compressed(save_path + '/mask_' + str(k), mask)
        print('data flip:{} saved'''.format(k))
        k +=1
        # else:
        # if ramdom_num1 > 0.0 and ramdom_num1 < 0.5:
        #         volume = flip(volume, 1)
        #         mask = flip(mask, 1)
        #         k += 1
        #         np.savez_compressed(save_path + '/volume_' + str(k + data_size), volume)
        #         np.savez_compressed(save_path + '/mask_' + str(k + data_size), mask)
        #         print('data flip:{} saved'''.format(k + data_size))

        # if ramdom_num2 > 0.0 and ramdom_num2 < 0.5:
        #     volume = zoom(volume, 0.8, -200)
        #     mask = zoom(mask, 0.8, 0)
        #
        # else:
        #     if ramdom_num3 > 0.0 and ramdom_num3 < 0.5:
        #         volume = zoom(volume, 1.2, -200)
        #         mask = zoom(mask, 1.2, 0)

        # feed_volume.append(volume)
        # feed_mask.append(mask)
        # end = time.time()
        # print('read data time', end - start)

    # return np.stack(feed_volume), np.stack(feed_mask)


def resize_image(input_path, output_path):
    dataset_num = int(len([name for name in os.listdir(input_path)]) / 2)
    print('dataset_contain: '+str(dataset_num)+'images and masks')
    for i in range(dataset_num):
        volume_1 = np.load(input_path + 'volume_' + str(i) + '.npz')

        mask_1 = np.load(input_path + 'mask_' + str(i) + '.npz')

        volume_1 = volume_1.f.arr_0 / 255
        mask_1 = mask_1.f.arr_0

        volume = np.zeros((224, 224, 3))
        mask = np.zeros((224, 224, 3))

        for k in range(3):
            volume[:, :, k] = resize(volume_1[:, :, k], (224, 224)) * 255
            mask[:, :, k] = resize(mask_1[:, :, k], (224, 224))

        if os.path.exists(output_path):
            np.savez_compressed(output_path + '/volume_' + str(i), volume)
            np.savez_compressed(output_path + '/mask_' + str(i), mask)
            print('image_and_mask' + str(i) + '_saved')
        else:
            os.makedirs(output_path)
            np.savez_compressed(output_path + '/volume_' + str(i), volume)
            np.savez_compressed(output_path + '/mask_' + str(i), mask)
            print('image_and_mask'+str(i)+'_saved')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    data_path = '/media/wang/Windows/train_data_3d_resize/'
    save_path = '/media/wang/Windows/train_data_3d_resize_elastic/'
    ofline_augument(data_path, save_path)
    # resize_image(data_path, save_path)