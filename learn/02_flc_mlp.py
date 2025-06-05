"""
PyTorch FLC (다층 퍼셉트론) 튜토리얼
=================================

이 튜토리얼에서는 PyTorch를 사용하여 다층 퍼셉트론(MLP)을 구현하고,
MNIST 숫자 이미지 분류 데이터셋에 적용합니다.

주요 단계:
1. 데이터 로드 및 전처리
2. 다층 퍼셉트론 (MLP) 모델 정의
3. 모델 훈련 (피팅)
4. 모델 평가
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 환경변수 로드
import os
from dotenv import load_dotenv
load_dotenv()

print("=== PyTorch FLC (다층 퍼셉트론) 튜토리얼 ===\n")

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
# DataLoader는 데이터를 미니 배치로 나누고, 셔플링, 병렬 로딩 등을 처리
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(f"훈련 데이터셋 크기: {len(train_dataset)}")
print(f"테스트 데이터셋 크기: {len(test_dataset)}\n")

# 2. 다층 퍼셉트론 (MLP) 모델 정의
print("2. 다층 퍼셉트론 (MLP) 모델 정의...")

class MLP(nn.Module):
    """
    간단한 다층 퍼셉트론(MLP) 모델
    
    다층 퍼셉트론은 여러 개의 완전 연결(Fully Connected) 레이어를 쌓아 만든 신경망입니다.
    각 층은 비선형 활성화 함수(여기서는 ReLU)를 사용하여 복잡한 패턴을 학습할 수 있습니다.
    배치 정규화와 드롭아웃을 사용하여 훈련 안정성을 높이고 과적합을 방지합니다.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        # 첫 번째 완전 연결층: 입력층에서 은닉층으로
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 배치 정규화 (Batch Normalization) 레이어:
        # 각 미니 배치마다 입력의 평균과 분산을 정규화하여 학습 과정을 안정화하고 수렴 속도를 높입니다.
        # 내부 공변량 변화(Internal Covariate Shift) 문제를 완화하고, 더 큰 학습률을 사용할 수 있게 합니다.
        self.bn1 = nn.BatchNorm1d(hidden_size)
        # 활성화 함수: ReLU
        self.relu = nn.ReLU()
        # 드롭아웃 (Dropout) 레이어:
        # 훈련 시 무작위로 일부 뉴런을 비활성화하여 모델이 특정 뉴런에 과도하게 의존하는 것을 방지합니다.
        # 이는 앙상블 효과를 주어 과적합을 줄이고 일반화 성능을 향상시킵니다. `p`는 드롭아웃할 뉴런의 비율입니다.
        self.dropout = nn.Dropout(p=0.5)
        
        # 두 번째 완전 연결층: 은닉층에서 출력층으로
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # --- 파라미터 초기화 (Parameter Initialization) ---
        # 신경망의 가중치와 편향을 적절히 초기화하는 것은 학습 안정성과 수렴 속도에 중요합니다.
        # ReLU 활성화 함수를 사용하는 레이어의 가중치는 Kaiming He 초기화(He normal/uniform)가 적합합니다.
        # 이는 ReLU의 특성상 음수 값을 0으로 만들기 때문에 발생하는 기울기 소실 문제를 완화하는 데 도움을 줍니다.
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu') # 출력층에도 ReLU가 없더라도 He 초기화를 사용하는 경우가 있습니다.
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        # --------------------------------------------------

    def forward(self, x):
        # 입력 이미지를 1차원 벡터로 평탄화 (28x28 = 784)
        x = x.view(x.size(0), -1) # x.size(0)은 배치 크기

        # 첫 번째 층 통과 -> 배치 정규화 -> 활성화 함수 -> 드롭아웃
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # 두 번째 층 통과
        out = self.fc2(out)
        return out

# 모델 파라미터 설정
input_size = 28 * 28  # MNIST 이미지 크기 (28x28)
hidden_size = 256     # 은닉층의 뉴런 수
num_classes = 10      # MNIST는 0~9까지 10개의 클래스

# 모델 인스턴스 생성
model = MLP(input_size, hidden_size, num_classes)

# GPU 사용 가능 시 디바이스 설정
# MPS (Metal Performance Shaders)는 Apple Silicon Mac에서 GPU 가속을 제공합니다.
# 이 코드는 MPS를 사용할 수 있다면 'mps' 디바이스를, 그렇지 않다면 'cpu' 디바이스를 선택합니다.
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

print(f"모델 구조:\n{model}")
print(f"모델을 {device} 장치로 이동했습니다.\n")

# 3. 모델 훈련 설정
print("3. 모델 훈련 설정...")

# 손실 함수: 교차 엔트로피 (CrossEntropyLoss) - 다중 분류에 적합
criterion = nn.CrossEntropyLoss()

# 옵티마이저: Adam
# `weight_decay` (L2 정규화): 과적합(overfitting)을 방지하는 정규화 기법 중 하나입니다.
# L2 정규화는 모델의 가중치(weights)가 너무 커지는 것을 제어하여 모델의 복잡도를 줄입니다.
# 손실 함수에 가중치 제곱합의 일정 비율을 더함으로써 큰 가중치에 패널티를 부여합니다.
# `weight_decay=0.0001`은 이 패널티의 강도를 조절하는 하이퍼파라미터입니다. 값이 너무 크면 모델이 충분히 학습되지 않을 수 있습니다.
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# 훈련 설정
num_epochs = 10

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
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device) # 이미지를 장치로 이동
        labels = labels.to(device) # 라벨을 장치로 이동
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파 및 최적화
        optimizer.zero_grad() # 기울기 초기화
        loss.backward()       # 역전파 (기울기 계산)
        optimizer.step()      # 파라미터 업데이트
        
        if (i+1) % 200 == 0:
            print (f'에포크 [{epoch+1}/{num_epochs}], 스텝 [{i+1}/{len(train_loader)}], 손실: {loss.item():.4f}')

print("훈련 완료!\n")

# 5. 모델 평가
print("5. 모델 평가...")

model.eval() # 모델을 평가 모드로 설정 (드롭아웃, 배치 정규화 등 비활성화)

with torch.no_grad(): # 기울기 계산 비활성화
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # 가장 높은 확률을 가진 클래스 선택
        total += labels.size(0) # 전체 샘플 수
        correct += (predicted == labels).sum().item() # 올바르게 예측한 샘플 수
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'테스트 정확도: {accuracy:.4f}')

# 개별 예측 예시
print("\n6. 개별 예측 예시...")

def imshow(img):
    img = img / 2 + 0.5     # 정규화 해제
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 테스트 데이터셋에서 이미지 하나 가져오기
example_image, example_label = test_dataset[0]

# 예측
model.eval()
with torch.no_grad():
    # 모델 입력 형태에 맞게 차원 확장 (배치 차원 추가)
    input_tensor = example_image.unsqueeze(0).to(device)
    output = model(input_tensor)
    _, predicted_class = torch.max(output.data, 1)

print(f"실제 라벨: {example_label}")
print(f"예측 라벨: {predicted_class.item()}\n")

print("=== 튜토리얼 완료 ===")
print("FLC(다층 퍼셉트론) 모델을 성공적으로 구현하고 훈련했습니다!")
print("주요 학습 내용:")
print("1. MNIST 데이터셋 로드 및 DataLoader 사용")
print("2. nn.Linear와 nn.ReLU를 사용한 MLP 모델 정의")
print("3. CrossEntropyLoss 손실 함수와 Adam 옵티마이저 사용")
print("4. 훈련 및 평가 루프 구현")
print("5. GPU를 활용한 모델 훈련")
