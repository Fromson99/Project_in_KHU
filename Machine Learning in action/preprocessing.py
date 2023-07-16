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


from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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


from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch

from torchvision import transforms

from PIL import Image

# 랜덤 시드 고정
np.random.seed(42)

# 시작 시간 기록
start_time = time.time()

# 이미지 로드 함수 정의
def load_img(filepath):
    img = cv2.imread(filepath)#이미지 파일 읽기 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#색상 공간 변화, 채널의 수가 변경 될 수 있다 BGR -> RGB로 변경
    return img

# DnCNN 모델 정의
class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_channels=64):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(3, num_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_channels, 3, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
# 커스텀 데이터셋 클래스 정의
class CustomDataset(data.Dataset):
    def __init__(self, noisy_image_paths, clean_image_paths, patch_size = 128, transform=None):
        self.clean_image_paths = [join(clean_image_paths, x) for x in listdir(clean_image_paths)]
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_paths[index])
        clean_image = load_img(self.clean_image_paths[index])

        H, W, _ = clean_image.shape

        
        # transform 적용
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        noisy_image_path = self.noisy_image_paths[index]
        


        return noisy_image, clean_image,noisy_image_path




# 하이퍼파라미터 설정
num_epochs = 90
batch_size = 1
learning_rate = 0.001
train_transform = Compose([
    ToTensor(),
    #Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# 데이터셋 경로
noisy_image_paths = './train_scan'
clean_image_paths = './train_clean'
train_dataset = CustomDataset(noisy_image_paths, clean_image_paths, transform=train_transform)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



def cutblur(im1, im2, prob=1.0, alpha=1.0):
    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
    print(ch,cw)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug

    return im1, im2


new_train_scan_path = './new_train_scan'

if not os.path.exists(new_train_scan_path):
    os.makedirs(new_train_scan_path)

count = 0
for noisy_images, clean_images,noisy_image_path in train_loader:
    HR_blur, LR = cutblur(clean_images, noisy_images)
    HR_blur_image = HR_blur
    HR_blur_image = HR_blur_image.cpu().squeeze(0)
    HR_blur_image = torch.clamp(HR_blur_image, 0, 1)  # 이미지 값을 0과 1 사이로 클램핑
    HR_blur_image = transforms.ToPILImage()(HR_blur_image)

    # Save denoised image
    output_filename = noisy_image_path[0]
    blur_filename = new_train_scan_path + '/' + output_filename.split('/')[-1][:-4] + '.tif'
    HR_blur_image.save(blur_filename) 
    print(f'Saved blur image: {blur_filename}')


