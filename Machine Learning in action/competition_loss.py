# -*- coding: utf-8 -*-
"""2023_Spring_KHU_MLIP_Competition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xDhXUz-80q8O-C1mboeISdMmyT3sbUud

# 기본 세팅
"""

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
import torchvision
import numpy as np
import torch
from sklearn.model_selection import KFold
# 랜덤 시드 고정
np.random.seed(42)



# 이미지 로드 함수 정의
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

"""# 모델 훈련"""
#content loss
class ContentLoss(nn.Module):
    def __init__(self, loss):
        super(ContentLoss, self).__init__()
        self.criterion = loss # L1, L2 선택
        self.net = self.content_model()

    def get_loss(self, pred, target):
        pred_f = self.net(pred)
        target_f = self.net(target)
        loss = self.criterion(pred_f, target_f)

        return loss

    def content_model(self):
        self.cnn = torchvision.models.vgg19(weights='DEFAULT').features
        self.cnn.cuda()
        # Content loss 계산을 위한 레이어 선택
        content_layers = ['relu_8']
        
        model = nn.Sequential()
        i = 0
        for layer in self.cnn.children():
        # Content loss 계산을 위한 모델 추출
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                break
        
        return model

#Nafnet
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
        
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), #gap,1x1xc
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 =nn.GroupNorm(1,c)
        self.norm2 = nn.GroupNorm(1,c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta #??이해안되넹

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class Nafnet(nn.Module):
  def __init__(self,img_channel=3, channel=32):
    super().__init__()
    self.intro = nn.Conv2d(img_channel,channel,kernel_size=3, padding=1, stride=1, groups=1,bias=True)
    self.block1 = NAFBlock(channel)
    self.down1 = nn.Conv2d(channel,channel*2,2,2)
    self.block2 = NAFBlock(channel*2)
    self.down2 = nn.Conv2d(channel*2,channel*4,2,2)
    self.block3 = NAFBlock(channel*4)
    self.down3 = nn.Conv2d(channel*4,channel*8,2,2)
    self.block4 = nn.Sequential(
                    *[NAFBlock(channel*8) for _ in range(10)]
                )

    self.down4 = nn.Conv2d(channel*8,channel*16,2,2)

    self.block5 = NAFBlock(channel*16)

    self.up1 = nn.Sequential(
                    nn.Conv2d(channel*16, channel*16*2, 1, bias=False),
                    nn.PixelShuffle(2)
    )
    self.block6 = NAFBlock(channel*8)
    self.up2 = nn.Sequential(
                    nn.Conv2d(channel*8, channel*8*2, 1, bias=False),
                    nn.PixelShuffle(2)
    )
    self.block7 = NAFBlock(channel*4)
    self.up3 = nn.Sequential(
                    nn.Conv2d(channel*4, channel*4*2, 1, bias=False),
                    nn.PixelShuffle(2)
    )
    self.block8 = NAFBlock(channel*2)
    self.up4 = nn.Sequential(
                    nn.Conv2d(channel*2, channel*2*2, 1, bias=False),
                    nn.PixelShuffle(2)
    )
    self.block9 = NAFBlock(channel)
    self.out = nn.Conv2d(channel,3,kernel_size=3, padding=1, stride=1, groups=1,bias=True)

  def forward(self,x):
    x = self.intro(x)

    x_block1 = self.block1(x)
    x = self.down1(x_block1)

    x_block2 = self.block2(x)
    x = self.down2(x_block2)

    x_block3 = self.block3(x)
    x = self.down3(x_block3)

    x_block4 = self.block4(x)
    x = self.down4(x_block4)

    x_block5 = self.block5(x) 

    x = self.up1(x_block5)
    x = x + x_block4
    x_block6 = self.block6(x)

    x = self.up2(x_block6)
    x = x + x_block3
    x_block7 = self.block7(x)

    x = self.up3(x_block7)
    x = x + x_block2
    x_block8 = self.block8(x)

    x = self.up4(x_block8)
    x = x + x_block1
    x_block9 = self.block9(x)

    out = self.out(x_block9)
    return out

# import gc
# gc.collect()
# torch.cuda.empty_cache()

# 하이퍼파라미터 설정
num_epochs = 150
batch_size = 32
learning_rate = 0.001

#모델 초기화
device = torch.device('cuda')
model = Nafnet()
model.load_state_dict(torch.load('best_model_sgd.pth'))
model.to(device)



criterion = nn.L1Loss()
content = ContentLoss(nn.L1Loss())

optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.9))
#optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum = 0.99)


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
        #index=index//len(self.noisy_image_paths)
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_paths[index])
        clean_image = load_img(self.clean_image_paths[index])

        H, W, _ = clean_image.shape

        # 이미지 랜덤 크롭
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        noisy_image = noisy_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        clean_image = clean_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        
        # transform 적용
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

# 데이터셋 경로
noisy_image_paths = './train_scan'
clean_image_paths = './train_clean'

# 데이터셋 로드 및 전처리
train_transform = Compose([
    ToTensor(),
    #Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #Normalize(mean=[0.7447, 0.7026, 0.6983], std=[0.2781, 0.2881, 0.2947])
])

# 커스텀 데이터셋 인스턴스 생성
train_dataset = CustomDataset(noisy_image_paths, clean_image_paths, transform=train_transform)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델 학습
model.train()
best_loss = 9999.0
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    for noisy_images, clean_images in train_loader:
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_images)

        #loss = criterion(outputs, noisy_images-clean_images)
        loss = criterion(outputs, noisy_images-clean_images)
        perceptual_loss = content.get_loss(outputs, noisy_images-clean_images)
        loss += 0.1 * perceptual_loss
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * noisy_images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 현재 epoch의 loss가 최소 loss보다 작으면 모델 갱신
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_model_sgd.pth')
        print(f"{epoch+1}epoch 모델 저장 완료")
    # 종료 시간 기록
    end_time = time.time()
    # 소요 시간 계산
    training_time = end_time - start_time
    # 시, 분, 초로 변환
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    # 결과 출력
    print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")
