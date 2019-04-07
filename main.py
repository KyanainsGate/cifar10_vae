import network
import cifar10_loader
import tensorflow as tf
import numpy as np

data_cfg = {
    'Path_to_save_cifar10_bin': './cifar-10-batches-bin/',
    'Load_file_num': 5,  # if Load_file_num=3, "data_batch_1.bin" to "data_batch_3.bin" will be used to training
    'Data_Augmentation_Ratio': 1,
    'Get_Images_Per_One_file': 100,
    'File_Id_To_visualise': 0,
}

network_cfg = {
    'img_size_list': [32, 32, 3],
    'latent_dim': 100,
    'log_path': './log/',
    'log_overwrite_save': True,
}

# If you use GU, please setting follow;
tf_config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0",  # specify GPU number
        allow_growth=True
    )
)


def train_vae(total_epoch=1000, batch_size=256, log_out_span=10, log_path=network_cfg['log_path']):
    # Data
    dataset = cifar10_loader.Cifar10(dirpath=data_cfg['Path_to_save_cifar10_bin'],
                                     data_number_for_train=data_cfg['Load_file_num'],
                                     get_samples_per_one_file=data_cfg['Get_Images_Per_One_file'],
                                     )
    X_train, X_test, T_train, T_test, N_train, N_test \
        = dataset.fetch_bin_to_tensor(data_argumantation_int=data_cfg['Data_Augmentation_Ratio'], reshape3d=True)

    # Define graph (tensorflow)
    vae = network.VAE(
        img_w_h_ch=network_cfg['img_size_list'],
        latent_dim=network_cfg['latent_dim'],
        log_path=network_cfg['log_path'],
        rm_exists_in_logdir=network_cfg['log_overwrite_save'],
    )

    # Iteration
    sess = tf.Session(config=tf_config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    train_loss_list = []
    learn_percent = 0.0
    saver = tf.train.Saver()
    log_writer = tf.summary.FileWriter(log_path, sess.graph)

    for epoch in range(total_epoch):
        print('epoch %d | ' % epoch, end='')
        sum_loss = 0
        perm = np.random.permutation(N_train)
        cnt = 0
        for i in range(0, N_train, batch_size):
            perm_batch = perm[i:i + batch_size]
            train_img = X_train[perm[i:i + batch_size]]
            batch_num = len(perm_batch)
            feed = {
                vae.img_plh: train_img,
                vae.real_batch_holder: batch_num,
            }
            _, loss = sess.run([vae.optimize, vae.cost], feed_dict=feed)
            sum_loss += np.mean(loss) * batch_size
            cnt += 1
        loss_in_epoch = sum_loss / N_train
        train_loss_list.append(loss_in_epoch)
        print('Train loss %.3f | ' % (loss_in_epoch))

        if epoch % log_out_span == 0:
            # Training summary
            train_sum = sess.run(vae.sum_train, feed_dict=feed)
            # Test summary
            test_feed = {
                vae.img_plh: train_img,
                vae.real_batch_holder: batch_num,
            }
            test_sum = sess.run(vae.sum_train, feed_dict=test_feed)
            saver.save(sess, log_path + 'graph1')  # save graph.meta,graph.index and so on ...
            log_writer.add_summary(train_sum, epoch)  # Write log to tensorboard of train state
            log_writer.add_summary(test_sum, epoch)  # same
            print('save')


if __name__ == '__main__':
    print('Training iter')
    train_vae()
