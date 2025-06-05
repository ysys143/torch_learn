# `learn/04_cnn.py` 설명

이 문서는 `learn/04_cnn.py` 파일에 구현된 간단한 합성곱 신경망(CNN) 예제의 세부 사항을 자세히 설명합니다. 이 스크립트는 CIFAR-10 데이터셋을 이용하여 이미지 분류 작업을 수행하며, 데이터 로딩부터 모델 정의, 훈련, 평가까지의 전 과정을 포함하고 있습니다.

## 1. 데이터 로드 및 전처리

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### 주요 특징
- **`transforms.ToTensor()`**: PIL 이미지를 `[0, 1]` 범위의 텐서로 변환합니다.
- **`transforms.Normalize`**: 각 채널의 평균과 표준편차를 이용해 정규화하여 학습을 안정화합니다.
- **`datasets.CIFAR10`**: 10가지 클래스가 포함된 32×32 컬러 이미지 데이터셋을 로드합니다. `download=True` 옵션으로 데이터가 없을 경우 자동으로 다운로드합니다.
- **`DataLoader`**: 미니 배치 단위로 데이터를 제공하며, `shuffle=True`로 훈련 데이터의 순서를 섞어 모델이 특정 순서에 과적합되는 것을 방지합니다.

## 2. 모델 정의 (`SimpleCNN` 클래스)

```python
class SimpleCNN(nn.Module):
    """간단한 CNN 모델"""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # 파라미터 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### 주요 특징
- **컨볼루션 계층(`nn.Conv2d`)**: 이미지의 공간적 패턴을 추출합니다. 첫 번째 계층은 입력 채널 3개(RGB 이미지)를 32개 특성 맵으로 변환합니다.
- **배치 정규화(`nn.BatchNorm2d`)**: 훈련 중 내부 활성값의 분포 변화를 완화하여 더 빠르고 안정적인 학습을 돕습니다.
- **활성화 함수(`nn.ReLU`)**: 비선형성을 제공하여 모델이 복잡한 패턴을 학습할 수 있게 합니다.
- **풀링(`nn.MaxPool2d`)**: 특성 맵의 크기를 절반으로 줄여 계산량을 감소시키고, 위치 변동에 대한 모델의 불변성을 향상시킵니다.
- **완전 연결층(`nn.Linear`)**: 추출된 특징을 이용해 최종 클래스 확률을 예측합니다. 여기서는 64×8×8 크기의 특성을 256차원으로 변환한 뒤 10개 클래스에 대해 로짓을 출력합니다.
- **드롭아웃(`nn.Dropout`)**: 0.5의 비율로 일부 뉴런을 비활성화하여 과적합을 방지합니다.
- **파라미터 초기화**: 컨볼루션 레이어는 `kaiming_uniform_`, 선형 레이어는 `xavier_uniform_`으로 초기화하여 학습 초기에 안정적인 기울기를 제공합니다.

## 3. 훈련 설정

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
num_epochs = 5
```

### 주요 특징
- **디바이스 선택**: GPU 사용 가능 시 `cuda`, 그렇지 않으면 `cpu`를 사용합니다.
- **손실 함수(`CrossEntropyLoss`)**: 다중 클래스 분류 문제에서 일반적으로 사용되는 손실 함수입니다.
- **옵티마이저(`Adam`)**: 모멘텀과 적응형 학습률을 활용하여 효율적으로 파라미터를 업데이트합니다.
- **`weight_decay`**: L2 정규화 항으로, 과적합을 줄이는 데 도움이 됩니다.
- **`num_epochs`**: 전체 데이터셋을 학습에 사용하는 횟수를 지정합니다.

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
1. `model.train()`으로 모델을 훈련 모드로 전환합니다. 이는 `BatchNorm`과 `Dropout`이 훈련용 동작을 하도록 설정합니다.
2. **미니 배치 반복**: `DataLoader`에서 배치를 가져와 입력과 라벨을 디바이스로 이동합니다.
3. **순전파**: 모델에 입력을 전달하여 출력(로짓)을 얻습니다.
4. **손실 계산**: 예측과 실제 라벨 사이의 교차 엔트로피 손실을 계산합니다.
5. **기울기 초기화**: `optimizer.zero_grad()`로 이전 단계에서 누적된 기울기를 초기화합니다.
6. **역전파**: `loss.backward()`를 호출하여 각 파라미터에 대한 기울기를 계산합니다.
7. **파라미터 업데이트**: `optimizer.step()`으로 계산된 기울기에 따라 파라미터를 갱신합니다.
8. 에포크가 끝날 때마다 손실 값을 출력해 학습 진행 상황을 확인합니다.

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
- **`model.eval()`**: 평가 모드로 전환하여 드롭아웃을 비활성화하고 배치 정규화가 누적된 통계를 사용하도록 합니다.
- **`torch.no_grad()`**: 그래디언트 계산을 비활성화하여 메모리 사용량을 줄이고 추론 속도를 높입니다.
- **정확도 계산**: 예측 결과와 실제 라벨을 비교하여 전체 정확도를 구합니다.

## 6. 마무리

위 스크립트는 CNN 구조의 기본적인 흐름을 보여주며, 하이퍼파라미터나 네트워크 구조를 변경하여 더 복잡한 실험으로 확장할 수 있습니다. 데이터 증강(`RandomCrop`, `HorizontalFlip` 등)을 추가하면 성능을 더욱 향상시킬 수 있습니다.

