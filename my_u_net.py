import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from data_process import read_train_data
from data_process import read_data
import os
from medpy import metric
from utility import loss, add_conv, add_deconv, pool_op


def get_u_net(input, bn, activation, with_bn_):
    conv1_1 = add_conv(input, 3, 3, 1, 1, 64, 'conv1_1', bn, activate=activation, with_bn=with_bn_)
    conv1_2 = add_conv(conv1_1, 3, 3, 1, 1, 64, 'conv1_2', bn=bn, activate=activation, with_bn=with_bn_)
    pool1 = pool_op(conv1_2, 2, 2, 2, 2, 'max_pool1')
    print([pool1.get_shape()[0].value, pool1.get_shape()[1].value,pool1.get_shape()[2].value, pool1.get_shape()[3].value])

    conv2_1 = add_conv(pool1, 3, 3, 1, 1, 128, 'conv2_1', bn=bn, activate=activation, with_bn=with_bn_)
    conv2_2 = add_conv(conv2_1, 3, 3, 1, 1, 128, 'conv2_2', bn=bn, activate=activation, with_bn=with_bn_)
    pool2 = pool_op(conv2_2, 2, 2, 2, 2, 'max_pool2')
    print([pool2.get_shape()[0].value, pool2.get_shape()[1].value, pool2.get_shape()[2].value, pool2.get_shape()[3].value])

    conv3_1 = add_conv(pool2, 3, 3, 1, 1, 256, 'conv3_1', bn=bn, activate=activation, with_bn=with_bn_)
    conv3_2 = add_conv(conv3_1, 3, 3, 1, 1, 256, 'conv3_2', bn=bn, activate=activation, with_bn=with_bn_)
    pool3 = pool_op(conv3_2, 2, 2, 2, 2, 'max_pool3')
    print([pool3.get_shape()[0].value, pool3.get_shape()[1].value, pool3.get_shape()[2].value, pool3.get_shape()[3].value])

    conv4_1 = add_conv(pool3, 3, 3, 1, 1, 512, 'conv4_1', bn=bn, activate=activation, with_bn=with_bn_)
    conv4_2 = add_conv(conv4_1, 3, 3, 1, 1, 512, 'conv4_2', bn=bn, activate=activation, with_bn=with_bn_)
    pool4 = pool_op(conv4_2, 2, 2, 2, 2, 'max_pool4')
    print([pool4.get_shape()[0].value, pool4.get_shape()[1].value, pool4.get_shape()[2].value, pool4.get_shape()[3].value])

    conv5_1 = add_conv(pool4, 3, 3, 1, 1, 1024, 'conv5_1', bn=bn, activate=activation, with_bn=with_bn_)
    conv5_2 = add_conv(conv5_1, 3, 3, 1, 1, 1024, 'conv5_2', bn=bn, activate=activation, with_bn=with_bn_)
    print([conv5_2.get_shape()[0].value, conv5_2.get_shape()[1].value, conv5_2.get_shape()[2].value, conv5_2.get_shape()[3].value])

    merge1 = tf.concat([add_deconv(conv5_2, 2, 2, 2, 2, bn, 'deconv1', activate=activation, with_bn=with_bn_), conv4_2], axis=3)
    conv6_1 = add_conv(merge1, 3, 3, 1, 1, 512, 'conv6_1', bn=bn, activate=activation, with_bn=with_bn_)
    conv6_2 = add_conv(conv6_1, 3, 3, 1, 1, 512, 'conv6_2', bn=bn, activate=activation, with_bn=with_bn_)
    print([conv6_2.get_shape()[0].value, conv6_2.get_shape()[1].value, conv6_2.get_shape()[2].value,
           conv6_2.get_shape()[3].value])

    merge2 = tf.concat([add_deconv(conv6_2, 2, 2, 2, 2, bn,'deconv2', activate=activation, with_bn=with_bn_), conv3_2], axis=3)
    conv7_1 = add_conv(merge2, 3, 3, 1, 1, 256, 'conv7_1', bn=bn, activate=activation, with_bn=with_bn_)
    conv7_2 = add_conv(conv7_1, 3, 3, 1, 1, 256, 'conv7_2', bn=bn, activate=activation, with_bn=with_bn_)
    print([conv7_2.get_shape()[0].value, conv7_2.get_shape()[1].value, conv7_2.get_shape()[2].value,
           conv7_2.get_shape()[3].value])

    merge3 = tf.concat([add_deconv(conv7_2, 2, 2, 2, 2, bn,'deconv3', activate=activation, with_bn=with_bn_), conv2_2], axis=3)
    conv8_1 = add_conv(merge3, 3, 3, 1, 1, 128, 'conv8_1', bn=bn, activate=activation, with_bn=with_bn_)
    conv8_2 = add_conv(conv8_1, 3, 3, 1, 1, 128, 'conv8_2', bn=bn, activate=activation, with_bn=with_bn_)
    print([conv8_2.get_shape()[0].value, conv8_2.get_shape()[1].value, conv8_2.get_shape()[2].value,
           conv8_2.get_shape()[3].value])

    merge4 = tf.concat([add_deconv(conv8_2, 2, 2, 2, 2, bn,'deconv4', activate=activation, with_bn=with_bn_), conv1_2], axis=3)
    conv9_1 = add_conv(merge4, 3, 3, 1, 1, 64, 'conv9_1', bn=bn, activate=activation, with_bn=with_bn_)
    conv9_2 = add_conv(conv9_1, 3, 3, 1, 1, 64, 'conv9_2', bn=bn, activate=activation, with_bn=with_bn_)
    print([conv9_2.get_shape()[0].value, conv9_2.get_shape()[1].value, conv9_2.get_shape()[2].value,
           conv9_2.get_shape()[3].value])

    output = add_conv(conv9_2, 3, 3, 1, 1, 1, 'conv10_1', bn, activate='sigmoid', with_bn=False)
    # print(output)
    # print([output.get_shape()[0].value, output.get_shape()[1].value, output.get_shape()[2].value,
    #        output.get_shape()[3].value])
    return output


def run_unet(train_data_path, test_data_path, tf_model_save_path, pic_save_path, loss_save_path, order = 'ordered',
             weighted = 'tversky', batch_size=16, epoch = 30, learning_rate = 1e-5, add_bn = True, activate = 'relu'):

    train_accuracy_set = []
    test_accuracy_set = []

    train_data_size = int(len([name for name in os.listdir(train_data_path)])/2)
    # test_data_size = int(len([name for name in os.listdir(test_data_path)])/2)

    epoch_steps = int(train_data_size / batch_size)
    random_arr = np.random.permutation(train_data_size)

    with tf.name_scope('input'):
        images = tf.placeholder(dtype=tf.float32, shape=[None, 240, 240, 1], name='input_images')
        masks = tf.placeholder(dtype=tf.float32, shape=[None, 240, 240, 1], name='input_masks')
        bn_switch = tf.placeholder(dtype=tf.bool, name='bn_switch')

    with tf.name_scope('output'):
        outputs = get_u_net(images, bn_switch, activation=activate, with_bn_=add_bn)

    with tf.name_scope('loss'):
        function_loss = loss(masks, outputs, weighted)
        # reg_variable = tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # reg_term = tf.contrib.layer.apply_regularization(reg_variable)
        # function_loss += reg_term

    with tf.name_scope('optimizer'):
        global_ = tf.Variable(0)
        optimizer = tf.train.exponential_decay(learning_rate, global_, epoch_steps, 0.90, False)
        train_step = tf.train.AdamOptimizer(optimizer).minimize(function_loss, global_step=global_)

    with tf.name_scope("results"):
        predicter = tf.to_int32(outputs[:, :, :, 0] > 0.5, name='predicter')
        # correct_pred = tf.equal(tf.argmax(outputs, 3), tf.argmax(masks, 3))

        # predicter = axis_max(outputs)
        # correct_pred = tf.equal(predicter, tf.argmax(masks, axis=3))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        # result = axis_max(outputs)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = time.time()
        for i in range(int(epoch*train_data_size/batch_size)):
            num = i % epoch_steps
            image_train, mask_train = read_train_data(num, train_data_path, batch_size, random_arr, order)
            sess.run(train_step, feed_dict={images: image_train, masks: mask_train, bn_switch:add_bn})
            # print(sess.run(outputs, feed_dict={images: images_train[0:1],
            #                                 masks: masks_train[0:1]}))
            if i % 30 == 0:
                train_loss,  out_put_map= sess.run((function_loss, predicter), feed_dict={images: image_train, masks: mask_train, bn_switch:False})
                end1 = time.time()
                time_cost = end1 - start
                out_put_map = out_put_map[0]
                binary_label = mask_train[0][:, :, 0]

                dc = metric.dc(out_put_map, binary_label)
                print('step:%i'%i, "train_loss:%.3f"%train_loss, "train_accuracy:%.3f"%dc, "time_cost:%5.3fs" % time_cost)

                train_accuracy_set.append(dc)

            if i % 300 == 0:

                train_loss_2, output_1 = sess.run((function_loss, predicter), feed_dict={images: image_train, masks: mask_train, bn_switch: False})

                gs = plt.GridSpec(1, 3)

                plt.figure(1)
                plt.subplot(gs[0, 0])
                plt.imshow(np.reshape(image_train[0], (240, 240)), cmap='gray')

                plt.subplot(gs[0, 1])
                plt.imshow(np.reshape(output_1[0], (240, 240)), cmap='gray')

                plt.subplot(gs[0, 2])
                plt.imshow(mask_train[0][:, :, 0], cmap='gray')

                # plt.show()
                plt.savefig(pic_save_path+'figure_train' + str(i))

                all_loss = []
                all_accuracy = []
                for k in range(96):
                        volume_test = read_data(k, 'Brain', test_data_path)
                        mask_test = read_data(k, 'mask', test_data_path)
                        test_loss, out_put_map_test = sess.run((function_loss, predicter), feed_dict={images: volume_test, masks: mask_test, bn_switch:False})

                        out_put_map_test = out_put_map_test[0]
                        binary_label_test = mask_test[0][:, :, 0]
                        dc_test = metric.dc(out_put_map_test, binary_label_test)

                        all_loss.append(test_loss)
                        all_accuracy.append(dc_test)
                        end2 = time.time()
                        time_cost2 = end2 - start
                batch_accuracy = np.mean(all_accuracy)
                loss_2 = np.mean(all_loss)
                test_accuracy_set.append(batch_accuracy)
                print('step:%i'%i, "test_loss:%.3f"%loss_2, "test_accuracy:%.3f"%batch_accuracy, "time_cost:%5.3fs" % time_cost2)

                image_2 = read_data(10, 'Brain', test_data_path)
                mask_2 = read_data(10, 'mask', test_data_path)

                output_2 = sess.run(predicter, feed_dict={images: image_2, masks: mask_2, bn_switch:False})

                gs = plt.GridSpec(1, 3)

                plt.figure(1)
                plt.subplot(gs[0, 0])
                plt.imshow(np.reshape(image_2, (240, 240)), cmap='gray')

                plt.subplot(gs[0, 1])
                plt.imshow(np.reshape(output_2, (240, 240)), cmap='gray')

                plt.subplot(gs[0, 2])
                plt.imshow(mask_2[0][:, :, 0], cmap='gray')

                # plt.show()
                plt.savefig(pic_save_path+'figure_'+str(i))

            if i % 3000 == 0:
                np.save(loss_save_path + 'test_accuracy.npy', test_accuracy_set)
                np.save(loss_save_path + 'train_accuracy.npy', train_accuracy_set)

            epoch_num = i / epoch_steps
            if i > 0:
                if i % (5*epoch_steps) == 0:
                    save_path = saver.save(sess, tf_model_save_path + 'model_epoch_' + str(epoch_num) + '.ckpt')
                    print("*** Model saved in file***: ", save_path + '_'+str(epoch_num), 'epoch_num %i'%int(epoch_num))

