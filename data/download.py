# Script to automatically download and cache dataset
# Usage: python download.py
#
# To learn more about the dataset, access:
# https://www.cityscapes-dataset.com/
import os
import sys
import pip


# Download and cache dataset
def main():
    pass


def download(name='cityscapes', path='data/downloads'):
    """Select one of the available and implemented datasets to download:
    name=any(['cityscapes', 'camvid', 'labelme'])
    """
    if name == 'cityscapes':
        download_cityscapes(path)
    else:
        raise NotImplementedError


def download_cityscapes(path='data/downloads'):
    if hasattr(pip, 'main'):
        pip.main(['install', 'cityscrapesscripts'])
    else:
        raise EnvironmentError("pip is not installed")


if __name__ == "__main__":
    main()
