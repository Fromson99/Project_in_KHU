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
# 하이퍼파라미터 설정
num_epochs = 50
batch_size = 64
learning_rate = 0.001

# 데이터셋 경로
noisy_image_paths = './train_scan'
clean_image_paths = './train_clean'

# 데이터셋 로드 및 전처리
train_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 커스텀 데이터셋 인스턴스 생성
train_dataset = CustomDataset(noisy_image_paths, clean_image_paths, transform=train_transform)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(train_loader)
# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DnCNN 모델 인스턴스 생성 및 GPU로 이동
model = DnCNN().to(device)

# 손실 함수와 최적화 알고리즘 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
model.train()
best_loss = 9999.0
for epoch in range(num_epochs):
    running_loss = 0.0
    for noisy_images, clean_images in train_loader:
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, noisy_images-clean_images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * noisy_images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 현재 epoch의 loss가 최소 loss보다 작으면 모델 갱신
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_dncnn_model.pth')
        print(f"{epoch+1}epoch 모델 저장 완료")

# 종료 시간 기록
end_time = time.time()

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

model = DnCNN()
model.load_state_dict(torch.load('best_dncnn_model.pth'))
model.eval()

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
    
    print(f'Saved denoised image: {denoised_filename}')

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

        print(y_values)

        # 파일 이름과 Y 채널 값을 CSV 파일에 작성
        writer.writerow([file_name[:-4], y_values])

print('CSV file created successfully.')