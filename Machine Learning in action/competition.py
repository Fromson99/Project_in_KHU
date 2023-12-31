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
import torch
import torch.nn.functional as F
from PIL import Image

# 랜덤 시드 고정
np.random.seed(42)



# 이미지 로드 함수 정의
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

"""# 모델 훈련"""

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
  def __init__(self,img_channel=3, channel=16,width=256):
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
num_epochs = 50
batch_size = 32
learning_rate = 0.00001

#모델 초기화
device = torch.device('cuda')
model = Nafnet()

model.to(device)


#criterion = nn.MSELoss()
criterion = nn.MSELoss()
#content = ContentLoss(nn.MSELoss())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


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
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
        resized_image = F.interpolate(outputs.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)
        loss = criterion(outputs, noisy_images-clean_images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * noisy_images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 현재 epoch의 loss가 최소 loss보다 작으면 모델 갱신
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_model2.pth')
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


"""# 모델 테스트 (추론)"""

import os
from os import listdir
from os.path import join, splitext
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose

# 랜덤 시드 고정
np.random.seed(42)

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



model = Nafnet()
model.load_state_dict(torch.load('best_model2.pth'))
model.to(device)
model.eval()


# 데이터셋 경로
noisy_data_path = './test_scan'
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

class CustomDatasetTest(data.Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)
    def __getitem__(self, index):
        
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(self.noisy_image_paths[index])
        
        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, noisy_image_path

test_transform = Compose([
    ToTensor(),
    #Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 데이터셋 로드 및 전처리
noisy_dataset = CustomDatasetTest(noisy_data_path, transform=test_transform)

# 데이터 로더 설정
noisy_loader = DataLoader(noisy_dataset, batch_size=1, shuffle=False)

# 이미지 denoising 및 저장
for noisy_image, noisy_image_path in noisy_loader:
    noisy_image = noisy_image.to(device)
    noise = model(noisy_image)

    denoised_image = noisy_image - noise
    
    # denoised_image를 CPU로 이동하여 이미지 저장
    denoised_image = denoised_image.cpu().squeeze(0)
    denoised_image = torch.clamp(denoised_image, 0, 1)  # 이미지 값을 0과 1 사이로 클램핑
    denoised_image = transforms.ToPILImage()(denoised_image)

    # Save denoised image
    output_filename = noisy_image_path[0]
    denoised_filename = output_path + '/' + output_filename.split('/')[-1][:-4] + '.png'
    denoised_image.save(denoised_filename) 
    
    #print(f'Saved denoised image: {denoised_filename}')

"""# 정답 csv 파일 생성"""

import os
import cv2
import csv
import numpy as np

folder_path = './output'
output_file = 'output.csv'

# 폴더 내 이미지 파일 이름 목록을 가져오기
file_names = os.listdir(folder_path)
file_names.sort()

# CSV 파일을 작성하기 위해 오픈
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image File', 'Y Channel Value'])

    for file_name in file_names:
        # 이미지 로드
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)

        # 이미지를 YUV 색 공간으로 변환
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Y 채널 추출
        y_channel = image_yuv[:, :, 0]

        # Y 채널을 1차원 배열로 변환
        y_values = np.mean(y_channel.flatten())

        #print(y_values)

        # 파일 이름과 Y 채널 값을 CSV 파일에 작성
        writer.writerow([file_name[:-4], y_values])

print('CSV file created successfully.')