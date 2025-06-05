"""
PyTorch VGG 네트워크 튜토리얼
===========================

이 예제에서는 `torchvision.models`의 VGG16 구조를 활용하여
CIFAR-10 이미지 분류를 수행하는 방법을 보여줍니다.

주요 단계:
1. 데이터 로드 및 전처리
2. VGG16 모델 준비
3. 모델 훈련 (피팅)
4. 모델 평가
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

print("=== PyTorch VGG 튜토리얼 ===\n")

# 1. 데이터 로드 및 전처리
print("1. CIFAR-10 데이터셋 로드 중...")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"훈련 데이터셋 크기: {len(train_dataset)}")
print(f"테스트 데이터셋 크기: {len(test_dataset)}\n")

# 2. VGG16 모델 준비
print("2. VGG16 모델 준비...")

model = models.vgg16(weights=None)
model.classifier[6] = nn.Linear(4096, 10)

# 파라미터 초기화
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"모델 구조:\n{model}\n")

# 3. 모델 훈련 설정
print("3. 모델 훈련 설정...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
num_epochs = 5

print(f"훈련 설정:")
print(f"- 에포크 수: {num_epochs}")
print(f"- 학습률: {optimizer.param_groups[0]['lr']}")
print(f"- 손실 함수: {criterion.__class__.__name__}")
print(f"- 옵티마이저: {optimizer.__class__.__name__}")
print(f"- L2 정규화 (Weight Decay): {optimizer.param_groups[0].get('weight_decay', 'N/A')}\n")

# 4. 모델 훈련 (피팅)
print("4. 모델 훈련 시작...")

model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"에포크 [{epoch+1}/{num_epochs}] 손실: {loss.item():.4f}")

print("훈련 완료!\n")

# 5. 모델 평가
print("5. 모델 평가...")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"테스트 정확도: {accuracy:.4f}")

print("=== 튜토리얼 완료 ===")
print("VGG16 모델을 사용한 이미지 분류 튜토리얼이 완료되었습니다!")
