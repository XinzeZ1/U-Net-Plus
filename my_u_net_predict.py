import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_process import read_data
from medpy import metric
import os

input_path = 'F:/data/test/'
tf_model_path = '/model/fcn-11-14/unet.ckpt'
meta_path = '/model/fcn-11-14/unet.ckpt.meta'
save_path = 'predict/unet-11-15'


saver = tf.train.import_meta_graph(meta_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Session() as sess:

    saver.restore(sess, tf_model_path)
    graph = tf.get_default_graph()
    bn_switch = graph.get_tensor_by_name('input/bn_switch:0')
    volume = graph.get_tensor_by_name('input/input_images:0')
    mask = graph.get_tensor_by_name('input/input_masks:0')
    predicter = graph.get_tensor_by_name('results/predicter:0')

    map_score = graph.get_tensor_by_name('output/conv10_1/scope:0')

    test_data_size = int(len([name for name in os.listdir(input_path)]) / 2)

    for i in np.arange(0, test_data_size):
        images = read_data(i, 'volume', input_path)
        mask = read_data(i, 'mask', input_path)
        map_score_1, predicter_1 = sess.run((map_score, predicter), feed_dict={volume: images, bn_switch:False})
        gs = plt.GridSpec(1, 3)


        plt.figure(1)
        plt.subplot(gs[0, 0])
        plt.imshow(np.reshape(map_score_1, (512, 512, 3)))
        plt.title('map_score')

        plt.subplot(gs[0, 1])
        plt.imshow(np.reshape(predicter_1, (512, 512)), cmap='gray')
        plt.title('predict')

        plt.subplot(gs[0, 2])
        plt.imshow(np.argmax(mask[0], axis=2), cmap='gray')
        plt.title('label')

        plt.savefig(save_path+'/map_score_pic/map_score_' + str(i))
        np.save(save_path+'/map_score/map_score_' + str(i)+'.npy', map_score_1[0])
        np.save(save_path+'/predict_result/predict_' + str(i)+'.npy', predicter_1[0])

        print('processing: '+str(i)+'/'+str(test_data_size))
