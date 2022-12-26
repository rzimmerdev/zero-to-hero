# Script to automatically download and cache dataset
# Usage: python downloader.py
#
# To learn more about the dataset, access:
# https://www.cityscapes-dataset.com/
import os
import pip
from urllib.request import urlretrieve


def download_dataset(name='cityscapes', path='downloads/downloads'):
    """Select one of the available and implemented downloads to download:
    name=any(['cityscapes', 'camvid', 'labelme'])
    """
    if name == 'cityscapes':
        download_cityscapes(path)
    elif name == "mnist":
        download_mnist(path)
    else:
        raise NotImplementedError


def download_mnist(path="downloads/mnist"):
    remote_files = {"train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"}
    if not os.path.exists(path):
        os.makedirs(path)

    for file in remote_files.keys():
        if os.path.exists(path + "/" + file):
            continue

        urlretrieve(remote_files[file], path + "/" + file)


def download_cityscapes(path='downloads/cityscapes'):
    if hasattr(pip, 'main'):
        pip.main(['install', 'cityscapesscripts'])
    else:
        raise EnvironmentError("pip is not installed")
    print("Which dataset do you want to download?")
    os.system("csDownload -l")
    ds_name = input()
    while ds_name not in ['gtFine_trainvaltest', 'gtFine_trainval', 'gtFine_test',
                          'leftImg8bit_trainvaltest', 'leftImg8bit_trainval', 'leftImg8bit_test']:
        print("Invalid dataset name. Please try again.")
        ds_name = input()
    os.system(f"csDownload {ds_name} -d {path}/{ds_name}")
