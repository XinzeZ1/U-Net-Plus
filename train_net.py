from my_u_net import run_unet


def train_net(train_data_path, test_data_path, tf_model_save_path, pic_save_path, accuracy_save_path,
              loss_function='dice', lr=3e-5, rounds=50, net='u_net'):

    if net == 'u_net':
        run_unet(train_data_path, test_data_path, tf_model_save_path, pic_save_path, accuracy_save_path,
                    weighted=loss_function, add_bn=True, learning_rate=lr, epoch=rounds)


if __name__ == '__main__':
    train_data_dir = './data/train_data/'
    test_data_dir = './data/test_data/'

    tf_model_dir = './model/'
    pic_save_dir = './pre/'
    accuracy_save_dir = './acc/'

    train_net(train_data_dir, test_data_dir, tf_model_dir, pic_save_dir, accuracy_save_dir,
              loss_function='generalized_dice', lr=3e-5, rounds=500, net='u_net')




