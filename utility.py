import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np


def tversky_loss(mask, output, alpha, beta):
    smooth = 1e-10

    output = tf.clip_by_value(output, 1e-10, 1.0)
    PG = tf.reduce_sum(tf.multiply(mask, output), axis=(1, 2))

    ones = tf.constant(1.0, dtype=tf.float32)
    a = alpha * tf.reduce_sum(tf.multiply((ones - mask), tf.pow(output, 2)), axis=(1, 2))
    b = beta * tf.reduce_sum(tf.multiply(tf.pow((ones - output), 2), mask), axis=(1, 2))

    tver_loss = tf.reduce_mean(PG / ((PG + a + b) + smooth))
    return 1 - tver_loss


def left_loss(output, mask):
        mask_liver = mask[:, :, :, 0:2]

        loss1 = generalized_tversky(mask_liver, output[0], 0.3, 0.7)
        # mask = tf.transpose(mask, perm=[0, 3, 1, 2])
        mask_lesion = mask[:, :, :, ::2]
        # print('liver_mask_shape:', np.shape(mask_liver), 'lesion_mask_shape:', np.shape(mask_lesion))

        # w = tf.reduce_sum(mask, axis=(0, 1, 2)) + 1e-10

        # lam = (w[1]-w[2]) / w[2]

        loss2 = generalized_tversky(mask_lesion, output[1], 0.3, 0.7)

        return loss1 + 10*loss2


def right_loss(output, mask):
    mask_liver = mask[:, :, :, 0:2]

    loss2 = generalized_tversky(mask_liver, output, 0.3, 0.7)
    return loss2


def new_loss(output, mask):
    loss2 = tf.cond(tf.reduce_sum(mask[:, :, :, 2]) > 0, lambda :left_loss(output, mask), lambda : right_loss(output[0], mask))
    return loss2


def jaccard_loss(mask, output):
    fore_ground = tf.reduce_sum(tf.multiply(mask, output), axis=(1, 2))
    cardianlity = tf.reduce_sum(mask, axis=(1, 2))
    back_ground = tf.reduce_sum(tf.multiply((1-mask), output), axis=(1, 2))

    jaccard = tf.reduce_mean(fore_ground / (cardianlity + back_ground))
    return 1 -jaccard


def dice_loss(mask, output):
    smooth = 1e-10

    output = tf.clip_by_value(output, 1e-10, 1.0)
    PG = tf.reduce_sum(tf.multiply(mask, output), axis=(1, 2))+ smooth
    p2 = tf.reduce_sum(tf.pow(output, 2), axis=(1, 2))
    g2 = tf.reduce_sum(mask, axis=(1, 2))

    dice_coe = tf.reduce_mean(2*PG / (p2 + g2 + smooth))
    return 1 - dice_coe


def re_weighted_cross_entropy(mask, output):
    batch_size = tf.shape(output)[0]
    output_re = tf.reshape(output, [batch_size, -1, 2])
    mask_re = tf.reshape(mask, [batch_size, -1, 2])

    w = tf.reduce_sum(mask, axis=(1, 2)) + 1e-10
    reverse_mask = 1 - mask
    w_2 = tf.reduce_sum(reverse_mask, axis=(1, 2)) + 1e-10
    weight = w_2 / w
    weight = tf.reshape(weight, [batch_size, 1, 2])
    # one = tf.constant([1.0], dtype=tf.float32)
    # weight = tf.concat((one, weight[1:]), axis=0)

    weight_map = tf.multiply(weight, mask_re)
    weight_map = tf.reshape(weight_map, [-1, 2])
    output_re = tf.reshape(output_re, [-1, 2])
    # print(weight_map.get_shape().as_list())
    re_wei_cross_entropy = tf.losses.sigmoid_cross_entropy(weight_map, output_re)

    # output_re = tf.reshape(output, [-1, 2])
    # mask_re = tf.reshape(mask, [-1, 2])
    # re_wei_cross_entropy = tf.reduce_mean(-tf.reduce_sum(mask_re * tf.log(tf.clip_by_value(output_re, 1e-10, 1.0)), axis=(1)))

    return re_wei_cross_entropy


# def re_weighted_cross_entropy_mean(mask, output):
#     re_weighted_cross_entropy_all = []
#     batch_size = output.get_shape()[0].value
#     for i in range(batch_size):
#         result = re_weighted_cross_entropy(mask[i], output[i])
#         re_weighted_cross_entropy_all.append(result)
#     return tf.reduce_mean(re_weighted_cross_entropy_all)


def generalized_tversky(mask, output, alpha, beta):
    smooth = 1e-10
    w = tf.reduce_sum(mask, axis=(1, 2)) + 1e-10
    reverse_mask = 1 - mask
    w_2 = tf.reduce_sum(reverse_mask, axis=(1, 2)) + 1e-10
    weight = w_2 / w

    # weight = 1 / w**2
    # one = tf.constant([1.0], dtype=tf.float32)
    # weight = tf.concat((one, weight[1:]), axis=0)

    output = tf.clip_by_value(output, 1e-10, 1.0)
    PG = tf.reduce_sum(tf.multiply(mask, output), axis=(1, 2)) + smooth

    ones = tf.constant(1.0, dtype=tf.float32)
    a = alpha * tf.reduce_sum(tf.multiply((ones - mask), tf.pow(output, 2)), axis=(1, 2))
    b = beta * tf.reduce_sum(tf.multiply(tf.pow((ones - output), 2), mask), axis=(1, 2))

    weighted_tver_loss = tf.reduce_mean(weight*PG /(weight*((PG + a + b) + smooth)))
    return 1 - weighted_tver_loss


def generalized_dice(mask, output):
    smooth = 1e-10
    w = tf.reduce_sum(mask, axis=(1, 2)) + 1e-10
    reverse_mask = 1 - mask
    w_2 = tf.reduce_sum(reverse_mask, axis=(1, 2)) + 1e-10
    weight = w_2 / w

    # weight = 1 / w ** 2
    # one = tf.constant([1.0], dtype=tf.float32)
    # weight = tf.concat((one, weight[1:]), axis=0)

    output = tf.clip_by_value(output, 1e-10, 1.0)
    PG = tf.reduce_sum(tf.multiply(mask, output), axis=(1, 2)) + smooth
    p2 = tf.reduce_sum(tf.pow(output, 2), axis=(1, 2))
    g2 = tf.reduce_sum(mask, axis=(1, 2))

    gen_dice_coe = tf.reduce_mean(2*weight*PG / (weight*(p2 + g2) + smooth))
    return 1 - gen_dice_coe


def loss(mask, output, weighted='cross_entropy'):

    if weighted == 'cross_entropy':
        loss_2 = re_weighted_cross_entropy(mask, output)
        return loss_2
    elif weighted == 'dice':
        loss_2 = dice_loss(mask, output)
        return loss_2
    elif weighted == 'tversky':
        loss_2 = tversky_loss(mask, output, 0.3, 0.7)
        return loss_2
    elif weighted == 'generalized_tversky':
        loss_2 = generalized_tversky(mask, output, 0.3, 0.7)
        return loss_2
    elif weighted == 'generalized_dice':
        loss_2 = generalized_dice(mask, output)
        return loss_2
    # elif weighted == 'new_loss':
    #     loss_2 = new_loss(output, mask)
    #     return loss_2


def selu(x, name):
    with ops.name_scope(name) as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def batch_normalization_layer(x, trainphase, scope_bn):
    with tf.name_scope(scope_bn):
        gmma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), trainable=True, name='gamma')
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), trainable=True, name='beta')
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def var_mean_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(trainphase, var_mean_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(x, mean, var, beta, gmma, 1e-3)
    return normed


def add_separable_conv(input, dp_kh, dp_kw, po_kh, po_kw, stride, n_out, channel_multiplier, name, bn, rate=None, activate='relu', with_bn = True):
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    n_in = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        depthwise_kernel = tf.get_variable(name=scope+'dp_kernel', shape=[dp_kh, dp_kw, n_in, channel_multiplier], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(), regularizer=regularizer)
        pointwise_kernel = tf.get_variable(name=scope + 'po_kernel', shape=[po_kh, po_kw, n_in*channel_multiplier, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(), regularizer=regularizer)
        separable_conv = tf.nn.separable_conv2d(input, depthwise_kernel, pointwise_kernel, stride, 'SAME', name=scope + 'separable_conv')

        b = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_out]), name=scope + 'bias')

        output = tf.nn.bias_add(separable_conv, b, name='score_map')

    if with_bn:
        output = batch_normalization_layer(output, trainphase=bn, scope_bn=scope + 'bn')

    if activate == 'relu':
        activation = tf.nn.relu(output, name=scope + 'relu')
        return activation

    elif activate == 'selu':
        activation = selu(output, 'selu')
        return activation


def add_conv(input, kh, kw, dh, dw, n_out, name, bn, activate='relu', with_bn = False):
    regularizer = tf.contrib.layers.l2_regularizer(0.001)
    n_in = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=scope+'kernel', shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(), regularizer=regularizer)
        conv = tf.nn.conv2d(input, kernel, strides=[1, dh, dw, 1], padding='SAME')
        b = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_out]), name='bias')

        output = tf.nn.bias_add(conv, b, name='score_map')

        if with_bn:
            output = batch_normalization_layer(output, trainphase=bn, scope_bn=name+'bn')
        else:
            output = output

        if activate == 'relu':
            activation = tf.nn.relu(output, name='scope')

        elif activate == 'selu':
            activation = selu(output, 'se')

        elif activate == 'sigmoid':
            activation = tf.nn.sigmoid(output, name='scope')

        return activation


def add_atrous_conv(input, kh, kw, rate, n_out, name, bn, activate='selu', with_bn = False):
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    n_in = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=scope+'kernel', shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(), regularizer=regularizer)
        conv = tf.nn.atrous_conv2d(input, kernel, rate=rate, padding='SAME')
        b = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_out]), name=scope+'bias')

        output = tf.nn.bias_add(conv, b, name='score_map')

        if with_bn:
            output = batch_normalization_layer(output, trainphase=bn, scope_bn=scope+'bn')
        else:
            output = output

        if activate == 'relu':
            activation = tf.nn.relu(output, name=scope+'relu')
            return activation

        elif activate == 'selu':
            activation = selu(output, 'selu')
            return activation

        elif activate == 'softmax':
            activation = tf.nn.softmax(output, name=scope+'softmax', axis=3)
            return activation


def add_deconv(input, kh, kw, dh, dw, bn, name, activate, with_bn = False):
    regularizer = tf.contrib.layers.l2_regularizer(0.001)
    shape_x = input.get_shape()
    shape = tf.shape(input)
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=scope + 'kernel', shape=[kh, kw, shape_x[3].value // 2, shape_x[3].value], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(), regularizer=regularizer)
        output_shape = tf.stack([shape[0], shape[1]*2, shape[2]*2, shape[3] // 2])
        deconv = tf.nn.conv2d_transpose(input, kernel, output_shape=output_shape, strides=[1, dh, dw, 1], padding='SAME')
        b = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[shape_x[3].value // 2]), name='bias')
        # print(deconv.get_shape().value)

        output = tf.nn.bias_add(deconv, b)

        if with_bn:
            output = batch_normalization_layer(output, trainphase=bn, scope_bn='bn')
        else:
            output = output

        if activate == 'relu':
            output = tf.nn.relu(output, name='scope_2')

        elif activate == 'selu':
            output = selu(output, 'selu')
        return output


def add_deconv_3(input, kh, kw, dh, dw, bn, name, activate, with_bn = False):
    regularizer = tf.contrib.layers.l2_regularizer(0.001)
    shape_x = input.get_shape()
    shape = tf.shape(input)
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=scope + 'kernel', shape=[kh, kw, shape_x[3].value // 2, shape_x[3].value], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(), regularizer=regularizer)
        output_shape = tf.stack([shape[0], shape[1]*2, shape[2]*2, 3])
        deconv = tf.nn.conv2d_transpose(input, kernel, output_shape=output_shape, strides=[1, dh, dw, 1], padding='SAME')
        b = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[3]), name='bias')
        # print(deconv.get_shape().value)

        output = tf.nn.bias_add(deconv, b)

        if with_bn:
            output = batch_normalization_layer(output, trainphase=bn, scope_bn='bn')
        else:
            output = output

        if activate == 'relu':
            output = tf.nn.relu(output, name='scope_2')

        elif activate == 'selu':
            output = selu(output, 'selu')
        return output


def pool_op(input, kh, kw, dh, dw, name):
    return tf.nn.max_pool(input, [1, kh, kw, 1], padding='SAME', strides=[1, dh, dw, 1], name=name)


def residual_block_for_unet(input, inner_channel, output_channel, train_scope, name):
    with tf.variable_scope(name):
        input_channel = input.get_shape().as_list()[-1]
        if 2*input_channel == output_channel:

            short_cut = tf.pad(input, [[0, 0], [0, 0], [0, 0], [input_channel//2, input_channel//2]])
            conv_1 = add_conv(input, 1, 1, 1, 1, inner_channel, name + 'conv_1', train_scope, 'relu', with_bn=True)

        elif input_channel == output_channel:
            short_cut = input
            conv_1 = add_conv(input, 1, 1, 1, 1, inner_channel, name+'conv_1', train_scope, 'relu', with_bn=True)

        else:
            conv_1 = add_conv(input, 1, 1, 1, 1, inner_channel, name + 'conv_1', train_scope, 'relu', with_bn=True)

            # conv_2 = add_conv(conv_1, 3, 3, 1, 1, inner_channel, name + 'conv_2', train_scope, 'relu', with_bn=True)
            conv_2 = add_separable_conv(conv_1, 3, 3, 1, 1, [1, 1, 1, 1], inner_channel, 1, 'sep_conv', train_scope, activate='relu', with_bn=True)

            conv_3 = add_conv(conv_2, 1, 1, 1, 1, output_channel, name + 'conv_2', train_scope, 'relu', with_bn=True)
            # print('1')
            return conv_3

        # conv_2 = add_conv(conv_1, 3, 3, 1, 1, inner_channel, name + 'conv_2', train_scope, 'relu', with_bn=True)
        conv_2 = add_separable_conv(conv_1, 3, 3, 1, 1, [1, 1, 1, 1], inner_channel, 1, 'sep_conv', train_scope, activate='relu', with_bn=True)

        conv_3 = add_conv(conv_2, 1, 1, 1, 1, output_channel, name + 'conv_2', train_scope, 'relu', with_bn=True)
        out_put = tf.add(short_cut, conv_3)
        return out_put


def residual_block_for_unet_atrous(input, inner_channel, output_channel, train_scope, name):
    with tf.variable_scope(name):
        input_channel = input.get_shape().as_list()[-1]

        if  output_channel == 2*input_channel:
            short_cut = tf.pad(input, [[0, 0], [0, 0], [0, 0], [input_channel//2, input_channel//2]])
            conv_1 = add_conv(input, 1, 1, 1, 1, inner_channel, name + 'conv_1', train_scope, 'selu', with_bn=False)
            conv_2 = add_atrous_conv(conv_1, 3, 3, 2, inner_channel, name + 'conv_2', train_scope, 'selu',
                                  with_bn=False)
            conv_3 = add_conv(conv_2, 1, 1, 1, 1, output_channel, name + 'conv_2', train_scope, 'selu',
                                  with_bn=False)
            out_put = tf.add(short_cut, conv_3)
            return out_put

        elif input_channel == output_channel:
            conv_1 = add_conv(input, 1, 1, 1, 1, inner_channel, name+'conv_1', train_scope, 'selu', with_bn=False)
            conv_2 = add_atrous_conv(conv_1, 3, 3, 2, inner_channel, name+'conv_2' ,train_scope, 'selu',with_bn=False)
            conv_3 = add_conv(conv_2, 1, 1, 1, 1, output_channel, name + 'conv_2', train_scope, 'selu', with_bn=False)
            short_cut = input
            out_put = tf.add(short_cut, conv_3)

            return out_put


def resize_and_conv(features, reshape_ratio, channel_ratio, name):

    new_height = features.get_shape()[1].value
    new_width = features.get_shape()[2].value
    channel = features.get_shape()[3].value

    up_features = tf.image.resize_nearest_neighbor(features, (reshape_ratio*new_height, reshape_ratio*new_width))
    # print('up_size:', up_features.get_shape().as_list())
    up = add_conv(up_features, 3, 3, 1, 1, channel//channel_ratio, name=name, bn=False)

    return up


def conv_op(input, name, kh, kw, output_channel, stride):
    regularizer = tf.contrib.layers.l2_regularizer(0.001)
    input_channel = input.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        kernel = tf.get_variable(name + 'kernel', [kh, kw, input_channel, output_channel],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d(), regularizer=regularizer)
        conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[output_channel]), trainable=True, name='bias')
        z = tf.nn.bias_add(conv, bias)
    return z