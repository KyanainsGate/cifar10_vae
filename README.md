# VAE & CNN example of CIFAR10(Tensorflow 1.1x )
It is a tutorial of a VAE (Variational AutoEncoder) and CNN(Convolutional Neural Network) with Tensorflow and python.  
For this tutorial, we use CIFAR-10.  
That is a common benchmark in machine learning for image recognition.

## CIFAR-10 is what ??

Like this.

---

![表紙](example.png)

---
The details are shown in the following URL.  
http://www.cs.toronto.edu/~kriz/cifar.html  


You can generate this figure by running [cifar10_loader.py](./cifar10_loader.py)  as follows;
```bash
$ python cifar10_loader.py
```

ということで，CIFAR-10でVAEによるreconstructやCNNによるclassificationをやるコードです．  
Windows10 + Pycharm なら，tensorboardはwsl上で起動しましょう．  
なぜなら，windowsがpermission周りにうるさかったりフォルダを手放さなかったりするから．

batch-normのコーディングや，cifar10のデータローダー，data argumentationの参考になるかな？
画像を上図みたいに並べるのも，探すの結構難航したから，参考になるかも（難航したのは自分だけ．．．．？）．  
~~就活でtensorflow書けるって言って歩ているのに，何も見せられるコードが無かったからコレ作りました．~~

## Environment
* OS
  * Windows10 (+ wsl for tensorboard)
  * Ubuntu 16.04
  
* Python packages and version
  * Python 3.6.xxx
  * tensorflow (tensorflow-gpu) 1.1xxx
  * numpy 1.14.xxx
  * opencv-python 1.14.xxx
  * Pillow 5.xxx

## Contents
This project contains the following codes.

| Code| Explanation |
| ------ | ------ |
| [main.py](./main.py)   | Train the model of VAE (by train_vae() ) and CNN-classification (by train_classification() )|
| [network.py](./network.py)  | The model of VAE(3-layer-encoder +3-layer-decoder) & CNN(5layer+batch_norm) |
| [cifar10_loader.py](./cifar10_loader.py)  | Download CIFAR10-binary-data automatically and some techniques of data-argumentation are implemented. |

## For Quick Start
Run VAE model (Default)
```bash
$ python main.py
```
If you want to run CNN classification model, comment out train_vae() in [main.py](./main.py) , and comment in train_classification().  
By the tensorboard, you can see the loss (or generated images) transition.  
```bash
$ cd <saved_dir>/VAEsample/<log_dir_name>
$ tensorboard --logdir=./ --port 6006
```
