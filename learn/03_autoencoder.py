"""
PyTorch Autoencoder (오토인코더) 튜토리얼
======================================

이 튜토리얼에서는 PyTorch를 사용하여 간단한 오토인코더를 구현하고,
MNIST 숫자 이미지 데이터셋을 사용하여 이미지의 차원 축소 및 재구성을 시연합니다.

주요 단계:
1. 데이터 로드 및 전처리
2. 오토인코더 모델 정의 (인코더, 디코더)
3. 모델 훈련 (피팅)
4. 원본 및 재구성된 이미지 시각화
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 환경변수 로드
import os
from dotenv import load_dotenv
load_dotenv()

print("=== PyTorch Autoencoder 튜토리얼 ===\n")

# 1. 데이터 로드 및 전처리
print("1. MNIST 데이터셋 로드 중...")

# 데이터 전처리: 이미지를 텐서로 변환하고 정규화
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 픽셀 값을 -1과 1 사이로 정규화
])

# 훈련 및 테스트 데이터셋 다운로드
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 데이터 로더 설정
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

print(f"훈련 데이터셋 크기: {len(train_dataset)}")
print(f"테스트 데이터셋 크기: {len(test_dataset)}\n")

# 2. 오토인코더 모델 정의
print("2. 오토인코더 모델 정의...")

class Autoencoder(nn.Module):
    """
    간단한 오토인코더 모델
    
    오토인코더는 입력 데이터를 압축(인코딩)하여 잠재 공간(Latent Space)의 표현으로 만들고,
    이 잠재 표현을 다시 원본 데이터 형태로 복원(디코딩)하는 신경망입니다.
    입력과 출력이 같아지는 것을 목표로 하며, 비지도 학습에 활용됩니다.
    배치 정규화와 드롭아웃을 사용하여 훈련 안정성을 높이고 과적합을 방지합니다.
    """
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Autoencoder, self).__init__()
        
        # 인코더 (Encoder): 입력 -> 잠재 공간
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size), # 배치 정규화: 학습 안정화 및 수렴 속도 향상
            nn.ReLU(),
            nn.Dropout(p=0.2), # 드롭아웃: 과적합 방지. 훈련 시에만 작동
            nn.Linear(hidden_size, latent_dim), # 잠재 공간의 차원
            nn.BatchNorm1d(latent_dim), # 배치 정규화
            nn.ReLU(),
            nn.Dropout(p=0.2) # 드롭아웃
        )
        
        # 디코더 (Decoder): 잠재 공간 -> 출력
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.BatchNorm1d(hidden_size), # 배치 정규화
            nn.ReLU(),
            nn.Dropout(p=0.2), # 드롭아웃
            nn.Linear(hidden_size, input_size),
            nn.Tanh() # 픽셀 값을 -1과 1 사이로 매핑하도록 Tanh 사용
        )

        # --- 파라미터 초기화 (Parameter Initialization) ---
        # 신경망의 가중치와 편향을 적절히 초기화하는 것은 학습 안정성과 수렴 속도에 중요합니다.
        # ReLU 활성화 함수를 사용하는 레이어의 가중치는 Kaiming He 초기화가 적합합니다.
        # Tanh 활성화 함수를 사용하는 레이어의 가중치는 Xavier/Glorot 초기화가 적합합니다.
        
        # 인코더 레이어 초기화
        nn.init.kaiming_uniform_(self.encoder[0].weight, nonlinearity='relu') # input_size -> hidden_size
        if self.encoder[0].bias is not None:
            nn.init.zeros_(self.encoder[0].bias)

        nn.init.kaiming_uniform_(self.encoder[4].weight, nonlinearity='relu') # hidden_size -> latent_dim
        if self.encoder[4].bias is not None:
            nn.init.zeros_(self.encoder[4].bias)

        # 디코더 레이어 초기화
        nn.init.kaiming_uniform_(self.decoder[0].weight, nonlinearity='relu') # latent_dim -> hidden_size
        if self.decoder[0].bias is not None:
            nn.init.zeros_(self.decoder[0].bias)
            
        # Tanh를 사용하는 마지막 레이어는 Xavier 초기화가 더 적합할 수 있습니다.
        nn.init.xavier_uniform_(self.decoder[4].weight) # hidden_size -> input_size (Tanh 활성화)
        if self.decoder[4].bias is not None:
            nn.init.zeros_(self.decoder[4].bias)
        # --------------------------------------------------

    def forward(self, x):
        # 입력 이미지를 1차원 벡터로 평탄화
        x = x.view(x.size(0), -1)
        
        # 인코딩
        encoded = self.encoder(x)
        
        # 디코딩
        decoded = self.decoder(encoded)
        
        return decoded

# 모델 파라미터 설정
input_size = 28 * 28  # MNIST 이미지 크기 (784)
hidden_size = 128     # 인코더/디코더의 중간층 뉴런 수
latent_dim = 32       # 잠재 공간의 차원 (원본보다 작게 설정)

# 모델 인스턴스 생성
model = Autoencoder(input_size, hidden_size, latent_dim)

# GPU 사용 가능 시 디바이스 설정
# MPS (Metal Performance Shaders)는 Apple Silicon Mac에서 GPU 가속을 제공합니다.
# 이 코드는 MPS를 사용할 수 있다면 'mps' 디바이스를, 그렇지 않다면 'cpu' 디바이스를 선택합니다.
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

print(f"모델 구조:\n{model}")
print(f"모델을 {device} 장치로 이동했습니다.\n")

# 3. 모델 훈련 설정
print("3. 모델 훈련 설정...")

# 손실 함수: MSELoss (Mean Squared Error) - 이미지 재구성이므로 픽셀 값 차이 최소화
criterion = nn.MSELoss()

# 옵티마이저: Adam
# `weight_decay` (L2 정규화): 과적합(overfitting)을 방지하는 정규화 기법 중 하나입니다.
# L2 정규화는 모델의 가중치(weights)가 너무 커지는 것을 제어하여 모델의 복잡도를 줄입니다.
# 손실 함수에 가중치 제곱합의 일정 비율을 더함으로써 큰 가중치에 패널티를 부여합니다.
# `weight_decay=0.0001`은 이 패널티의 강도를 조절하는 하이퍼파라미터입니다. 값이 너무 크면 모델이 충분히 학습되지 않을 수 있습니다.
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# 훈련 설정
num_epochs = 20
losses = []

print(f"훈련 설정:")
print(f"- 에포크 수: {num_epochs}")
print(f"- 학습률: {optimizer.param_groups[0]['lr']}")
print(f"- 손실 함수: {criterion.__class__.__name__}")
print(f"- 옵티마이저: {optimizer.__class__.__name__}")
print(f"- L2 정규화 (Weight Decay): {optimizer.param_groups[0].get('weight_decay', 'N/A')}\n")

# 4. 모델 훈련 (피팅)
print("4. 모델 훈련 시작...")

model.train() # 모델을 훈련 모드로 설정

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, images.view(images.size(0), -1)) # 출력과 원본 이미지 비교
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'에포크 [{epoch+1}/{num_epochs}], 스텝 [{i+1}/{len(train_loader)}], 손실: {loss.item():.4f}')
    
    losses.append(loss.item())

print("훈련 완료!\n")

# 5. 원본 및 재구성된 이미지 시각화
print("5. 원본 및 재구성된 이미지 시각화...")

model.eval() # 모델을 평가 모드로 설정

with torch.no_grad():
    # 테스트 데이터셋에서 몇 개 샘플 가져오기
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    
    # 원본 이미지를 오토인코더에 통과시켜 재구성
    reconstructed_images = model(images).cpu()
    
    # 원본 이미지 (정규화 해제)
    original_images_display = images.cpu().view(images.size(0), 1, 28, 28) / 2 + 0.5
    # 재구성된 이미지 (정규화 해제)
    reconstructed_images_display = reconstructed_images.view(images.size(0), 1, 28, 28) / 2 + 0.5
    
    # 첫 10개 이미지 시각화
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))
    for images, row in zip([original_images_display, reconstructed_images_display], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img.numpy()), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
    plt.savefig('autoencoder_reconstruction.png', dpi=150, bbox_inches='tight')
    print("원본 및 재구성된 이미지 결과가 'autoencoder_reconstruction.png'로 저장되었습니다.\n")

# 손실 그래프 시각화
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title('오토인코더 훈련 손실 (Autoencoder Training Loss)')
plt.xlabel('에포크 (Epoch)')
plt.ylabel('손실 (Loss)')
plt.grid(True)
plt.savefig('autoencoder_loss.png', dpi=150, bbox_inches='tight')
print("손실 그래프가 'autoencoder_loss.png'로 저장되었습니다.\n")

print("=== 튜토리얼 완료 ===")
print("오토인코더 모델을 성공적으로 구현하고 훈련했습니다!")
print("주요 학습 내용:")
print("1. MNIST 데이터셋 로드 및 DataLoader 사용")
print("2. nn.Linear와 nn.ReLU, nn.Tanh를 사용한 오토인코더 모델 정의")
print("3. MSELoss 손실 함수와 Adam 옵티마이저 사용")
print("4. 훈련 루프 구현 및 이미지 재구성 시각화")
print("5. GPU를 활용한 모델 훈련 (가능 시)")
