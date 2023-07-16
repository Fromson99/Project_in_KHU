import random
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from os.path import join
from os import listdir
import time
import zipfile




train_scan_zip = zipfile.ZipFile('./test_scan/test_scan.zip')
train_scan_zip.extractall('./test_scan')
train_scan_zip.close()