# my pc config
import matplotlib
matplotlib.use('TkAgg')

# For graphical module
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Related CIFAR 10
import os
import sys
import urllib.request
import tarfile

"""
Functions for Data-Argumantation
"""


def _horizontal_flip(image, rate=0.5):
    """
    Horizontal flip

    :param image: The tensor (=ndarray) of image.shape of it is (batch, h, w, ch)
    :param rate: The probability whether arg-images are processed of NOT.
    :return: Processed (or NOT) image, flag(DONE means 1, NOT DONE means 0).
    """
    flag_h = 0
    if np.random.rand() < rate:
        image = image[:, :, ::-1, :]
        # image = image.reshape([batch_size, :, ::-1, :])
        flag_h = 1
    return image, flag_h


def _vertical_flip(image, rate=0.5):
    """
    Vertical flip

    :param image: The tensor (=ndarray) of image.shape of it is (batch, h, w, ch)
    :param rate: The probability whether  arg images are processed of NOT.
    :return: Processed (or NOT) image tensor, flag(DONE means 1, NOT DONE means 0)
    """
    flag_v = 0
    if np.random.rand() < rate:
        image = image[:, ::-1, :, :]
        flag_v = 1
    return image, flag_v


def _random_crop(image, crop_size=(28, 28)):
    """

    :param image: The tensor (=ndarray) of image.shape of it is (batch, h, w, ch)
    :param crop_size: The size of processed images.
    :return: Processed (or NOT) image tensor, flag(DONE means 1, NOT DONE means 0)
    """
    _, h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    bottom = top + crop_size[0]
    right = left + crop_size[1]

    image = image[:, top:bottom, left:right, :]
    return image


def _cutout(image_origin, mask_size=12):
    """
    Mask the random areas of images
    Paper : Improved Regularization of Convolutional Neural Networks with Cutout
    URL : https://arxiv.org/abs/1708.04552

    :param image_origin: The tensor (=ndarray) of image.shape of it is (batch, h, w, ch)
    :param mask_size: The length of one side of mask area.
    :return: Image tensor,
    """
    image = np.copy(image_origin)
    batch_size = image.shape[0]
    mask_value = image.mean() - np.sum(np.arange(batch_size)) / batch_size  # 0-9の数字の平均
    _, h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    top_list = np.random.randint(low=0 - mask_size // 2, high=h - mask_size, size=batch_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    left_list = np.random.randint(low=0 - mask_size // 2, high=w - mask_size, size=batch_size)
    bottom = top + mask_size
    bottom_list = top_list + mask_size
    right = left + mask_size
    right_list = left_list + mask_size
    top_list[top_list < 0] = 0
    left_list[left_list < 0] = 0
    # image[:, top:bottom, left:right, :].fill(mask_value)
    for j in range(batch_size):
        image[j, top_list[j]:bottom_list[j], left_list[j]:right_list[j], :].fill(mask_value)
    return image


def comb_reflec_and_cutput(image, rate=0.5):
    """
    Combine all Data-augmentation functions based on arg-probability

    :param image:The tensor (=ndarray) of image.shape of it is (batch, h, w, ch)
    :param rate:The probability whether arg-images are processed of NOT.
    :return: images
    """
    image, flag1 = _horizontal_flip(image)
    image, flag2 = _vertical_flip(image)
    total_flag = flag1 + flag2
    if total_flag != 0:
        if np.random.rand() < rate:
            image = _cutout(image, mask_size=np.random.randint(low=12, high=20))
        else:
            pass
    else:
        image = _cutout(image, mask_size=np.random.randint(low=12, high=20))
    return image


class Cifar10:
    """
    Image were decoded by binary file, the shape of which (batch, channel, height(?), width) at first.
    It was converted to (batch, height, width, channel) and after that flattened for Data-Argumantation.
    It can decoded by follow;
        img = img.reshape(data_num, -1) #  ("-1"  == h * w * ch)
        img = img.reshape(-1, 32, 32, 3)

    """

    def __init__(self, cifar10_cfg):
        # Make new directory
        dirpath =  cifar10_cfg['Path_to_save_cifar10_bin']
        data_number_for_train = cifar10_cfg['Load_file_num']
        get_samples_per_one_file = cifar10_cfg['Get_Images_Per_One_file']

        if not os.path.exists(dirpath):
            print('CIFAR10 was NOT FOUND, so cretate new directory...')
            new_dirpath = dirpath.split('/cifar-10-batches-bin/')[0]
            os.makedirs(new_dirpath, exist_ok=True)
            url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
            filename = url.split('/')[-1]
            filepath = os.path.join(new_dirpath, filename)

            def _progress(cnt, chunk, total):
                now = cnt * chunk
                if (now > total): now = total
                sys.stdout.write('\rdownloading {} {} / {} ({:.1%})'.format(filename, now, total, now / total))
                sys.stdout.flush()

            urllib.request.urlretrieve(url, filepath, _progress)
            tarfile.open(filepath, 'r:gz').extractall(dirpath)

        self.cifar10_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Merge <data_number_for_train> separated training data (If False, all(=5)data will be merged)
        if data_number_for_train == False:
            self.data_number_for_train = 5
        else:
            self.data_number_for_train = data_number_for_train

        self.label_size = 1
        self.img_size = 32 * 32 * 3
        self.data_size = self.label_size + self.img_size
        # Merge <get_samples_per_one_file> images in each file (If False, all(=10000)data will be merged)
        if get_samples_per_one_file == False:
            self.get_samples_per_one_file = 10000
        else:
            self.get_samples_per_one_file = get_samples_per_one_file
        self.file_train_list = []
        self.file_test = ''
        self.dirpath = dirpath + '/cifar-10-batches-bin/'
        self.data_number_for_train = data_number_for_train
        self.file_train_list = [self.dirpath + 'data_batch_' + str(data_number) + '.bin' for data_number in
                                range(1, 1 + data_number_for_train)]
        self.file_test = self.dirpath + 'test_batch' + '.bin'

        # Fetch関数のreturn
        self.X_train = None
        self.X_test = None
        self.t_train = None
        self.t_test = None
        self.N_train = None
        self.N_test = None

        # Pre-Fetch Variables
        self.integrated_img_arr_list = None
        self.integrated_img_label_list = None
        self.img_w_h_ch_test = None
        self.label_test = None

    def fetch_bin_to_tensor(self, normalization=True, one_hot_vector=True, data_argumantation_int=1, reshape3d=False):
        print('Merging train sets || ', end='')
        if data_argumantation_int > 1.0:
            print('with DATA ARGUMANTATION ....', end='')
        # Training data
        for j in range(len(self.file_train_list)):
            path = self.file_train_list[j]
            with open(path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8,
                                     count=self.data_size * self.get_samples_per_one_file)
            label = np.zeros(self.get_samples_per_one_file)
            img_arr = np.zeros([self.get_samples_per_one_file, self.img_size])
            # (32 * 32 * 3 * <image num>, 10)．
            for i in range(self.get_samples_per_one_file):
                start = i * self.data_size
                label[i] = data[start]
                img_arr[i] = data[start + 1: start + self.data_size]
            img = img_arr.reshape(self.get_samples_per_one_file, 3, 32, 32).transpose(0, 2, 3,
                                                                                      1)  # データ数の行，縦ｘ横ｘチャンネルが列として，32x32X3(RGB)
            img_origin = img
            label_origin = label
            # argumantation process#
            if data_argumantation_int > 1:
                for i in range(1, data_argumantation_int):
                    new_img = comb_reflec_and_cutput(img_origin)
                    new_label = label_origin
                    img = np.r_[img, new_img]
                    label = np.r_[label, new_label]
                    # print(img.shape)
                    # print(label.shape)
            # argumantation process
            img_w_h_ch = img.reshape(data_argumantation_int * self.get_samples_per_one_file, -1)
            print('#', end='')
            # Merge
            if j == 0:
                self.integrated_img_arr_list = img_w_h_ch
                # print(img_w_h_ch.shape)
                self.integrated_img_label_list = label
                # print(label.shape)
            else:
                self.integrated_img_arr_list = np.vstack((img_w_h_ch, self.integrated_img_arr_list))
                self.integrated_img_label_list = np.hstack((label, self.integrated_img_label_list))
        print('|| Completed')
        # Test data
        print('Set Test data (The ratio of TRAIN : TEST is ' + str(self.data_number_for_train) + ' : 1 )', end='')
        with open(self.file_test, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8,
                                 count=self.data_size * self.get_samples_per_one_file)  # data_sizeでサイズを指定．BGRそれぞれ32 x 32と，そのラベル
        img_arr_test = np.zeros([self.get_samples_per_one_file, self.img_size])  # 横一列に32x32x3がずらーっと．10列ならぶ．
        self.label_test = np.zeros(self.get_samples_per_one_file)
        for i in range(self.get_samples_per_one_file):
            start = i * self.data_size
            self.label_test[i] = data[start]  # i個目の要素に対し，ラベルを格納．32x32x3が始まる直前にクラス名が出るのがcifar10の仕様
            img_arr_test[i] = data[start + 1: start + self.data_size]  # 行列に格納
        img_test = img_arr_test.reshape(self.get_samples_per_one_file, 3, 32, 32).transpose(0, 2, 3,
                                                                                            1)  # データ数の行，列として，32x32X3(RGB)
        self.img_w_h_ch_test = img_test.reshape(self.get_samples_per_one_file, -1)
        print('Completed')

        # normalization for CNN
        self.X_train = self.integrated_img_arr_list
        self.X_test = self.img_w_h_ch_test
        self.T_train = self.integrated_img_label_list
        self.T_test = self.label_test
        # Data size
        self.N_train = self.X_train.shape[0]
        self.N_test = self.X_test.shape[0]

        if normalization == True:
            # Normalization of images
            self.X_train = self.X_train / 255.
            self.X_test = self.X_test / 255.
        if one_hot_vector == True:
            # convert to one-hot-vector
            self.T_train = np.eye(10)[self.T_train.astype("int")]
            self.T_test = np.eye(10)[self.T_test.astype("int")]

        if reshape3d is False:
            return self.X_train, self.X_test, self.T_train, self.T_test, self.N_train, self.N_test
        else:
            return self.X_train.reshape(-1, 32, 32, 3), self.X_test.reshape(-1, 32, 32,
                                                                            3), self.T_train, self.T_test, self.N_train, self.N_test

    def test_figure(self, dataSet_number_1to5):
        # 10枚show
        path = self.file_train_list[dataSet_number_1to5 - 1]
        data_num = 10
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, count=self.data_size * data_num)
        label = np.zeros(data_num)
        img_arr = np.zeros([data_num, self.img_size])  # 横一列に32x32x3がずらーっと．10列ならぶ．
        for i in range(data_num):
            start = i * self.data_size  # 32x32x3-1個目の要素からざーっとデータを取得
            label[i] = data[start]  # i個目の要素に対し，ラベルを格納．32x32x3が始まる直前にクラス名が出るのがcifar10の仕様
            img_arr[i] = data[start + 1: start + self.data_size]  # 行列に格納
        img = img_arr.reshape(data_num, 3, 32, 32).transpose(0, 2, 3, 1)  # データ数の行，列として，32x32X3(RGB)
        img = img.reshape(data_num, -1)
        img = img.reshape(data_num, 32, 32, 3)
        canvas = Image.new('RGB', (320, 175), (
            240, 240, 240))  # 新規の画像(マージに利用する背景画像)を作成し，canvasとして保存, サイズが(320,175)で，(240, 240, 240)は背景の色
        draw = ImageDraw.Draw(canvas)  # ImageDraw() などで画像オブジェクトを処理する
        for i in range(data_num):  # 10回のループ
            num = i if i < 5 else i - 5  # 列番号をここで指定．5列目到達で次の6枚目を0としてカウント
            x = 20 + (32 + 20) * num  # マージン20pixelから，少し空けて
            y = 20 if i < 5 else 20 + 32 + 45  # マージンと画像と文字列用のスペース(45)
            canvas.paste(Image.fromarray(np.uint8(img[i])), (x, y))  # (x, y)で中心のがぞうをはじく
            draw.text((x, y + 32 + 10), self.cifar10_names[int(label[i])], fill='#000000')
        canvas.save('sample' + str(dataSet_number_1to5) + '.png')
        canvas.show()


if __name__ == '__main__':
    # Example of hot to use
    print('-cifar10_loader.py exapmle -')

    data_cfg = {
        'Path_to_save_cifar10_bin': './cifar-10-batches-bin/',
        'Load_file_num': 5,  # if Load_file_num=3, "data_batch_1.bin" to "data_batch_3.bin" will be used to training
        'Data_Augmentation_Ratio': 1,
        'Get_Images_Per_One_file': 10,
    }
    File_Id_To_visualise = 2

    dataset = Cifar10(data_cfg)
    X_train, X_test, T_train, T_test, N_train, N_test \
        = dataset.fetch_bin_to_tensor(data_argumantation_int=1, reshape3d=True)
    dataset.test_figure(File_Id_To_visualise)
    plt.show()
