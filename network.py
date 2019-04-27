import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import os
import shutil

SEED_NUMPY = 71
np.random.seed(SEED_NUMPY)
xavier_initializer = tf.contrib.layers.xavier_initializer()


# VAE module
class VAE(object):
    def __init__(self, net_cfg):

        # Graph definition
        self.img_size = net_cfg['img_shape']
        self.restore = False
        if 'restore' in net_cfg:
            self.restore = net_cfg['restore']
        self.latent_dim = net_cfg['latent_dim']
        self.lr = 0.001
        if 'learning_rate' in net_cfg:
            self.lr = net_cfg['learning_rate']
        self.log_path = net_cfg['log_path']
        self.rm_exists_in_logdir = False
        if 'log_overwrite_save' in net_cfg:
            self.rm_exists_in_logdir = net_cfg['log_overwrite_save']
        # Tneosrflow's plh
        self.img_plh = tf.placeholder(dtype=tf.float32,
                                      shape=(None, self.img_size[0], self.img_size[1], self.img_size[2]),
                                      name="inpt_img")
        self.is_training_holder = tf.placeholder(dtype=bool, name='is_training')
        self.real_batch_holder = tf.placeholder(dtype=tf.int32, shape=())
        # Graph building
        self.z_mean, self.z_log_var = self.encoder_net(self.img_plh, latent_dim=self.latent_dim)
        self.sampled_z = self._sample_z(self.real_batch_holder,
                                        self.latent_dim,
                                        self.z_mean,
                                        self.z_log_var)  # Re-parametrization trick
        self.decoder = self._decoder_net(self.sampled_z)

        self.cast_img_inpt = tf.cast(255.0 * self.img_plh, tf.uint8)
        self.cast_img_otpt = tf.cast(255.0 * self.decoder, tf.uint8)

        self.cost, self.latent_cost = self._loss_func(self.decoder, self.img_plh, self.z_mean, self.z_log_var)
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        self.sum_train, self.sum_test = self._create_summary(self.cost, self.latent_cost)
        # log configuration
        if self.rm_exists_in_logdir or not self.restore:
            if not os.path.exists(self.log_path):
                os.mkdir(self.log_path)
            else:
                shutil.rmtree(self.log_path)
        else:
            pass
        self.saver = tf.train.Saver()
        self.iter_counter = 0

    def encoder_net(self, inpt_holder, latent_dim, reuse=False):
        with tf.variable_scope('Encoder', reuse=reuse):
            conv1 = tf.layers.conv2d(inputs=inpt_holder,
                                     filters=3,
                                     kernel_size=4,
                                     strides=2,
                                     padding='same',
                                     kernel_initializer=xavier_initializer,
                                     activation=tf.nn.relu)

            # Convolution outputs [batch, 8, 8, 64]
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=128,
                                     kernel_size=4,
                                     strides=2,
                                     padding='same',
                                     kernel_initializer=xavier_initializer,
                                     activation=tf.nn.relu)

            # Convolution outputs [batch, 4, 4, 64]
            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=64,
                                     kernel_size=4,
                                     strides=2,
                                     padding='same',
                                     kernel_initializer=xavier_initializer,
                                     activation=tf.nn.relu)

            flat = tf.contrib.layers.flatten(conv3)
            z_mean = tf.layers.dense(flat, units=latent_dim, name='z_mean')
            z_log_var = tf.layers.dense(flat, units=latent_dim, name='z_log_var')
            return z_mean, z_log_var  # var = sigma ** 2

    def _decoder_net(self, inpt_holder, reuse=False):
        with tf.variable_scope('Decoder', reuse=reuse):
            z_develop = tf.layers.dense(inpt_holder, units=4 * 4 * 64)
            net = tf.nn.relu(tf.reshape(z_develop, [-1, 4, 4, 64]))
            net = tf.layers.conv2d_transpose(inputs=net,
                                             filters=64,
                                             kernel_size=4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=xavier_initializer,
                                             activation=tf.nn.relu)

            # Transposed convolution outputs [batch, 16, 16, 64]
            net = tf.layers.conv2d_transpose(inputs=net,
                                             filters=128,
                                             kernel_size=4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=xavier_initializer,
                                             activation=tf.nn.relu)

            # Transposed convolution outputs [batch, 32, 32, 3]
            net = tf.layers.conv2d_transpose(inputs=net,
                                             filters=3,
                                             kernel_size=4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=xavier_initializer)

            net = tf.nn.sigmoid(net)

            return net

    def _sample_z(self, batch_dim, sample_dim, z_mean, z_log_var):
        """
        Given latent mean and log_var, sample latent values for decode

        :param batch_dim:
        :param sample_dim:
        :param z_mean:
        :param z_log_var:
        :return:
        """
        samples = tf.random_normal([batch_dim, sample_dim], 0, 1, dtype=tf.float32)
        z = z_mean + (tf.exp(0.5 * z_log_var) * samples)
        return z

    def generate_from_gausian(self, batch_dim):
        """
        output values by trained-decoder from gaussian noise

        :param batch_dim: A string; the number of generation images.
        :return: A 4-D Tensor of; [batch_dim, w, h, ch]
        """
        samples = tf.random_normal([batch_dim, self.latent_dim], 0, 1, dtype=tf.float32)
        x = self._decoder_net(samples, reuse=True)
        return x

    def _loss_func(self, pred_tensor, true_tensor, z_mean, z_log_var):
        # Compute KL divergence (latent loss)
        D_kl = -.5 * tf.reduce_sum(1. + z_log_var - tf.pow(z_mean, 2) - tf.exp(z_log_var),
                                   reduction_indices=1, name='latent_loss')

        inferenc_vec = tf.contrib.layers.flatten(pred_tensor)
        true_vec = tf.contrib.layers.flatten(true_tensor)
        # MSE loss
        # reconstruction_loss = tf.reduce_sum(0.5 * (true_vec - inferenc_vec) ** 2, name='reconst_loss')
        # Cross entropy loss
        reconstruction_loss = -tf.reduce_sum(true_vec * tf.log(tf.clip_by_value(inferenc_vec, 1e-10, 1.0)) + \
                                             (1 - true_vec) * tf.log(tf.clip_by_value(1 - inferenc_vec, 1e-10, 1.0)), 1,
                                             name='reconstruction_loss_RGB')
        reconst_scalar = tf.reduce_mean(reconstruction_loss)
        latent_scalar = tf.reduce_mean(D_kl)
        cost = reconst_scalar + latent_scalar
        return cost, latent_scalar

    def _create_summary(self, cost_, latent_cost):
        with tf.name_scope("summary/Train/") as scope:
            train1 = tf.summary.scalar('Loss', cost_)
            train2 = tf.summary.scalar('latent_loss', latent_cost)
            train_true_img = tf.summary.image('In', self.cast_img_inpt, max_outputs=3)
            train_pred_img = tf.summary.image('Out', self.cast_img_otpt, max_outputs=3)
            summary_train = tf.summary.merge([train1, train2, train_true_img, train_pred_img])
        with tf.name_scope("summary/Test/") as scope:
            test1 = tf.summary.scalar('Loss', cost_)
            test2 = tf.summary.scalar('latent_loss', latent_cost)
            test_true_img = tf.summary.image('In', self.cast_img_inpt, max_outputs=3)
            test_pred_img = tf.summary.image('Out', self.cast_img_otpt, max_outputs=3)
            summary_eval = tf.summary.merge([test1, test2, test_true_img, test_pred_img])
        return summary_train, summary_eval


class CNN(object):
    def __init__(self, net_cfg):
        # Graph definition
        self.img_size = net_cfg['img_shape']
        self.restore = False
        if 'restore' in net_cfg:
            self.restore = net_cfg['restore']
        self.lr = 0.001
        if 'learning_rate' in net_cfg:
            self.lr = net_cfg['learning_rate']
        self.log_path = net_cfg['log_path']
        self.rm_exists_in_logdir = False
        if 'log_overwrite_save' in net_cfg:
            self.rm_exists_in_logdir = net_cfg['log_overwrite_save']
        # Tneosrflow's plh
        self.img_plh = tf.placeholder(dtype=tf.float32,
                                      shape=(None, self.img_size[0], self.img_size[1], self.img_size[2]),
                                      name="inpt_img")
        self.t_plh = tf.placeholder(tf.float32, [None, 10])
        self.is_training_holder = tf.placeholder(dtype=bool, name='is_training')
        self.real_batch_holder = tf.placeholder(dtype=tf.int32, shape=())
        # Build Graph
        self.y = self.cnn_net(self.img_plh, self.is_training_holder)
        self.loss = self._loss_func(self.y, self.t_plh)
        self.accuracy = self._accuracy(self.y, self.t_plh)
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sum_train, self.sum_test = self._create_summary(self.loss, self.accuracy)
        # log configuration
        if self.rm_exists_in_logdir or not self.restore:
            if not os.path.exists(self.log_path):
                os.mkdir(self.log_path)
            else:
                shutil.rmtree(self.log_path)
        else:
            pass
        self.saver = tf.train.Saver()
        self.iter_counter = 0

    def cnn_net(self, inpt_holder, is_training, reuse=False):
        with tf.variable_scope('CNN', reuse=reuse):
            x = tf.layers.conv2d(inpt_holder, filters=64, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=96, kernel_size=[3, 3], strides=[2, 2], padding='SAME')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=128, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=256, kernel_size=[3, 3], strides=[2, 2], padding='SAME')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)
            x = tf.contrib.layers.flatten(x)
            out = tf.layers.dense(x, 10)
            # print(out)
            return out

    def _loss_func(self, nn_output, label_onehot):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=label_onehot, logits=nn_output)
        cost = tf.reduce_mean(xentropy)
        return cost

    def _accuracy(self, nn_output, label_onehot):
        correct_prediction = tf.equal(tf.argmax(nn_output, 1), tf.argmax(label_onehot, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _create_summary(self, cost_, accuracy_):
        with tf.name_scope("summary/Train/") as scope:
            train1 = tf.summary.scalar('Loss', cost_)
            train2 = tf.summary.scalar('Accuracy', accuracy_)
            summary_train = tf.summary.merge([train1, train2])
        with tf.name_scope("summary/Test/") as scope:
            test1 = tf.summary.scalar('Loss', cost_)
            test2 = tf.summary.scalar('Accuracy', accuracy_)
            summary_eval = tf.summary.merge([test1, test2])
        return summary_train, summary_eval


if __name__ == '__main__':
    # Example of hot to use
    print('-network.py exapmle -')
    network_cfg = {
        'img_shape': [32, 32, 3],
        'latent_dim': 64,
        'log_path': './test/',
        'log_overwrite_save': True,
    }
    vae_net = VAE(network_cfg)
    # classifier_net = CNN(network_cfg)
