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
    def __init__(self, img_w_h_ch, latent_dim, lr=10e-3, log_path='./log/',
                 restore=False,
                 rm_exists_in_logdir=False):
        # Graph definition
        self.img_size = img_w_h_ch
        self.restore = restore
        self.latent_dim = latent_dim
        self.lr = lr
        self.log_path = log_path
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
        if rm_exists_in_logdir or not restore:
            if not os.path.exists(self.log_path):
                shutil.rmtree(self.log_path)
                os.mkdir(self.log_path)
        else:
            pass
        self.saver = tf.train.Saver()
        self.iter_counter = 0

        #

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
        with tf.variable_scope('cmd_decoder', reuse=reuse):
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
        samples = tf.random_normal([batch_dim, self.latent_dim], 0, 1, dtype=tf.float32)
        x = self._decoder_net(samples, reuse=True)
        return x

    def _loss_func(self, inferenc_vec, true_vec, z_mean, z_log_var):
        # Compute KL divergence (latent loss)
        D_kl = -.5 * tf.reduce_sum(1. + z_log_var - tf.pow(z_mean, 2) - tf.exp(z_log_var),
                                   reduction_indices=1, name='latent_loss')
        reconstruction_loss = tf.reduce_sum(0.5 * (true_vec - inferenc_vec) ** 2, name='reconst_loss')
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


def show_normalized_img_square(normalized_img_batch, generate_num=100, w_mergin=5, h_mergin=5, w_init_mergin=5, \
                               h_init_mergin=5, background_color=(15, 15, 15), save_fig_name='', perm=False,
                               imshow_mode=True):
    """
    Draw the images of DNN's output on canvas and show (or save).
     Require : PIL(If not, please install pillows by pip)

    :param normalized_img_batch: The tensor of NN output, the shape of it is (batch, h, w, channel=3),
            the value of them have to be normalized (e.g. 0 <= pixel_value <= 1)
    :param generate_num: the generate image num
    :param w_mergin:
    :param h_mergin:
    :param w_init_mergin:
    :param h_init_mergin:
    :param background_color:
    :param save_fig_name: If you specify it, the <save_fig_name> is saved (e.g.) "path_to_dir/out.png"
    :param perm:
    :param imshow_mode: If true, plt.show will called,
    :return:
    """

    """
    
    Args : normalized_img_batch = ndarray of normalized image(0 <= pixel <=1)
        normalized_img_batch.shape = (batch, width, shape, channel=3)
    """
    gen_img_vertical = int(np.sqrt(generate_num))
    img = normalized_img_batch * 255
    print('Set normalized image shape : ', normalized_img_batch.shape)
    # Set cavans for image
    img_batch = img.shape[0]
    img_w = img.shape[1]
    img_h = img.shape[2]
    canvas_w = w_init_mergin * 2 + img_w * gen_img_vertical + w_mergin * (gen_img_vertical - 1)
    canvas_h = h_init_mergin * 2 + img_h * gen_img_vertical + h_mergin * (gen_img_vertical - 1)
    canvas = Image.new('RGB', (canvas_w, canvas_h), background_color)
    print('Image_type:RGB batch_size:' + str(generate_num) + ' width:' + str(canvas_w) + ' height' + str(canvas_h))
    draw = ImageDraw.Draw(canvas)
    if perm:
        perm_id = np.random.permutation(img_batch)
        print('Pemutation is True')
    print('Drawing ...')
    for i in range(generate_num):
        row = i % gen_img_vertical
        column = int(i / gen_img_vertical)
        x = w_init_mergin + (img_w + w_mergin) * row  # マージン20pixelから，少し空けて
        y = h_init_mergin + (img_h + h_mergin) * column
        if perm:
            i = perm_id[i]
        canvas.paste(Image.fromarray(np.uint8(img[i])), (x, y))  # (x, y)を起点として画像を描く
    if not save_fig_name == '':
        print('save path to .png : ' + save_fig_name)
        canvas.save(save_fig_name)
    if imshow_mode:
        canvas.show()


if __name__ == '__main__':
    # Example of hot to use
    print('-network.py exapmle -')
    vae_net = VAE(
        img_w_h_ch=[32, 32, 3],
        latent_dim=20,
        log_path='./log/',
        rm_exists_in_logdir=True,
    )
