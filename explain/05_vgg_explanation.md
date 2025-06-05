# `learn/05_vgg.py` 설명

이 문서는 `learn/05_vgg.py` 파일에 구현된 VGG16 기반 이미지 분류 예제를 자세히 다룹니다. CIFAR-10 데이터셋을 224×224 크기로 변환하여 VGG 구조에 입력하고, 학습 과정에서 가중치를 직접 초기화하여 모델을 훈련합니다.

## 1. 데이터 로드 및 전처리

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### 주요 특징
- **이미지 리사이즈**: `transforms.Resize((224, 224))`로 CIFAR-10(32×32)을 VGG가 기대하는 224×224 크기로 변환합니다.
- **`transforms.Normalize`**: RGB 채널별 평균과 표준편차로 정규화하여 학습을 안정화합니다.
- **배치 크기**: VGG 모델은 파라미터 수가 많으므로 `batch_size=32`로 설정했습니다.

## 2. VGG16 모델 준비

```python
model = models.vgg16(weights=None)
model.classifier[6] = nn.Linear(4096, 10)
```

### 주요 특징
- **사전 학습 가중치 미사용**: `weights=None`으로 설정하여 임의로 초기화된 VGG16을 사용합니다.
- **클래스 수 조정**: CIFAR-10은 10개의 클래스로 구성되어 있으므로, 마지막 선형층을 `nn.Linear(4096, 10)`으로 교체합니다.

### 파라미터 초기화

```python
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
```

- **`kaiming_uniform_`**: ReLU와 함께 사용할 때 효과적인 초기화 방법으로, 깊은 네트워크에서 기울기 흐름을 원활하게 합니다.
- **`xavier_uniform_`**: 선형층 초기화에 많이 사용되며, 입력과 출력의 균형을 맞춰 학습 초기에 안정성을 제공합니다.

## 3. 훈련 설정

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
num_epochs = 5
```

### 주요 특징
- **디바이스 선택**: GPU가 있으면 속도 향상을 위해 `cuda`를 사용하고, 없으면 `cpu`에서 실행합니다.
- **학습률**: VGG 구조는 파라미터가 많으므로 학습률을 `0.0001`로 낮게 설정해 안정적인 학습을 유도합니다.
- **`weight_decay`**: 과적합을 완화하기 위해 0.0005의 L2 정규화를 적용합니다.

## 4. 모델 훈련 (피팅)

```python
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
```

### 단계별 설명
1. `model.train()`으로 훈련 모드를 활성화합니다.
2. **미니 배치 순회**: `train_loader`에서 데이터를 받아와 GPU/CPU로 전송합니다.
3. **순전파**: 이미지 배치를 모델에 입력하여 로짓을 계산합니다.
4. **손실 계산**: `CrossEntropyLoss`를 사용해 예측과 실제 라벨의 차이를 측정합니다.
5. **기울기 초기화 후 역전파**: `optimizer.zero_grad()`로 기울기를 초기화한 뒤, `loss.backward()`로 역전파합니다.
6. **파라미터 업데이트**: `optimizer.step()`을 호출하여 가중치를 업데이트합니다.
7. 에포크가 끝날 때마다 현재 손실 값을 출력합니다.

## 5. 모델 평가

```python
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
```

### 평가 단계 설명
- **`model.eval()`**: 평가 모드로 전환하여 드롭아웃을 비활성화하고 배치 정규화가 저장된 통계를 사용합니다.
- **`torch.no_grad()`**: 그래디언트를 계산하지 않아 메모리를 절약하고 추론 속도를 높입니다.
- **정확도 계산**: 전체 테스트 데이터에 대한 정확도를 측정하여 모델의 일반화 성능을 확인합니다.

## 6. 결론

이 예제는 사전 학습 없이 VGG16을 처음부터 훈련하는 기본적인 흐름을 보여 줍니다. 필요에 따라 데이터 증강을 추가하거나 학습률 스케줄러를 적용해 성능을 개선할 수 있습니다.

