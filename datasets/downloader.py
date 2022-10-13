# Script to automatically download and cache dataset
# Usage: python downloader.py
#
# To learn more about the dataset, access:
# https://www.cityscapes-dataset.com/
import os
import sys
import pip


# Download and cache dataset
def main():
    pass


def download(name='cityscapes', path='datasets/downloads'):
    """Select one of the available and implemented datasets to download:
    name=any(['cityscapes', 'camvid', 'labelme'])
    """
    if name == 'cityscapes':
        download_cityscapes(path)
    else:
        raise NotImplementedError


def download_cityscapes(path='datasets/downloads'):
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


if __name__ == "__main__":
    main()
