# `learn/02_flc_mlp.py` 설명

이 문서는 `learn/02_flc_mlp.py` 파일에 구현된 다층 퍼셉트론(MLP) 모델의 핵심 구성 요소인 모델 정의, 훈련 (피팅), 그리고 추론 (평가) 과정을 상세히 설명합니다.

## 1. 모델 정의 (`MLP` 클래스)

다층 퍼셉트론(MLP) 모델은 PyTorch의 `nn.Module`을 상속받아 정의됩니다. 이 모델은 여러 개의 완전 연결(Fully Connected) 레이어를 쌓아 만든 신경망으로, MNIST 숫자 이미지 분류와 같은 복잡한 패턴 학습에 사용됩니다.

```python
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
        # `forward` 메서드는 모델이 입력 `x`를 받았을 때 어떻게 계산을 수행할지 정의합니다.
        # 1. `x.view(x.size(0), -1)`: 입력 이미지는 (배치 크기, 채널, 높이, 너비) 형태입니다.
        #    이를 (배치 크기, 픽셀 수) 형태의 1차원 벡터로 평탄화합니다. 예를 들어, (64, 1, 28, 28) -> (64, 784).
        x = x.view(x.size(0), -1) 

        # 첫 번째 층 통과 -> 배치 정규화 -> 활성화 함수 -> 드롭아웃
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # 두 번째 층 통과
        out = self.fc2(out)
        return out
```

### 주요 특징:
-   **다층 구조**: 여러 개의 `nn.Linear` 레이어를 사용하여 입력 데이터를 점진적으로 변환하고 추상화합니다.
-   **`nn.ReLU` 활성화 함수**: 각 선형 레이어 사이에 `ReLU`를 적용하여 모델의 비선형성을 높이고 복잡한 패턴을 학습할 수 있게 합니다.
-   **`nn.BatchNorm1d` (배치 정규화)**: 각 레이어의 입력을 정규화하여 내부 공변량 변화를 줄이고 학습을 안정화하며 수렴 속도를 높입니다. 이는 더 높은 학습률을 사용할 수 있게 돕고 모델의 일반화 성능을 향상시킵니다.
-   **`nn.Dropout` (드롭아웃)**: 훈련 시 무작위로 뉴런의 일부를 비활성화하여 모델이 특정 특성이나 뉴런에 과도하게 의존하는 것을 방지합니다. 이는 앙상블 효과를 주어 과적합을 줄이고 모델의 일반화 성능을 높이는 데 효과적입니다.
-   **이미지 평탄화**: `forward` 메서드 내에서 `x.view()`를 사용하여 다차원 이미지 데이터를 1차원 벡터로 변환합니다. 이는 완전 연결층의 입력 요구사항에 맞추기 위함입니다.
-   **`nn.Module` 상속**: PyTorch 모델의 표준이며, 파라미터 관리 및 `forward`/`backward` 메서드 정의를 용이하게 합니다.
-   **파라미터 초기화 (Kaiming He)**: 모델의 가중치와 편향을 학습 시작 전에 특정 값으로 설정하는 과정입니다. ReLU 활성화 함수를 사용하는 레이어에는 `Kaiming He` 초기화가 적합하며, 이는 학습 초기에 기울기 소실이나 폭주 문제를 완화하여 모델이 안정적으로 학습되도록 돕습니다.

## 2. 모델 훈련 (피팅)

모델 훈련 과정은 정의된 `MLP` 모델이 MNIST 훈련 데이터셋에 맞춰 최적의 파라미터(가중치와 편향)를 학습하도록 하는 단계입니다.

```python
# 모델 인스턴스 생성
input_size = 28 * 28  # MNIST 이미지 크기 (28x28)
hidden_size = 256     # 은닉층의 뉴런 수
num_classes = 10      # MNIST는 0~9까지 10개의 클래스

model = MLP(input_size, hidden_size, num_classes)

# GPU 사용 가능 시 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 손실 함수: 교차 엔트로피 (CrossEntropyLoss) - 다중 분류에 적합
# `nn.CrossEntropyLoss()`는 다중 클래스 분류 문제에서 사용되는 일반적인 손실 함수입니다.
# 이 함수는 내부적으로 `softmax` 활성화 함수와 음의 로그 우도(Negative Log Likelihood)를 결합하여 계산합니다.
# 따라서 모델의 마지막 레이어에 `softmax`를 직접 추가할 필요가 없습니다.
criterion = nn.CrossEntropyLoss()

# 옵티마이저: Adam
# `optim.Adam`은 모델의 파라미터를 업데이트하여 손실 함수를 최소화하는 알고리즘입니다.
# `model.parameters()`는 훈련 가능한 모델의 모든 파라미터를 옵티마이저에 전달합니다.
# `lr=0.001`은 학습률(learning rate)로, 각 업데이트 단계에서 파라미터가 얼마나 크게 변경될지를 결정합니다.
# `weight_decay` (L2 정규화): 과적합(overfitting)을 방지하는 정규화 기법 중 하나입니다.
# L2 정규화는 모델의 가중치(weights)가 너무 커지는 것을 제어하여 모델의 복잡도를 줄입니다.
# 손실 함수에 가중치 제곱합의 일정 비율을 더함으로써 큰 가중치에 패널티를 부여합니다.
# `weight_decay=0.0001`은 이 패널티의 강도를 조절하는 하이퍼파라미터입니다. 값이 너무 크면 모델이 충분히 학습되지 않을 수 있습니다.
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# 훈련 설정
num_epochs = 10

# 모델 훈련 (피팅) 시작
model.train() # 모델을 훈련 모드로 설정합니다. 이는 Dropout이나 Batch Normalization과 같은 특정 레이어의 동작에 영향을 미칩니다.

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # `images.to(device)`, `labels.to(device)`: 데이터를 CPU에서 GPU로 이동합니다 (GPU 사용 가능 시).
        # 이는 훈련 속도를 크게 향상시킬 수 있습니다.
        images = images.to(device)
        labels = labels.to(device)
        
        # 순전파
        # `model(images)`: 훈련 이미지를 모델에 입력하여 예측 `outputs`를 얻습니다.
        # `criterion(outputs, labels)`: 모델의 예측과 실제 라벨을 비교하여 손실을 계산합니다.
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파 및 최적화
        # `optimizer.zero_grad()`: 이전 스텝에서 계산된 기울기를 0으로 초기화합니다.
        # `loss.backward()`: 현재 손실에 대한 모델 파라미터들의 기울기를 계산합니다.
        # `optimizer.step()`: 계산된 기울기를 사용하여 모델의 파라미터를 업데이트합니다.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 진행 상황 출력 (선택 사항)
        # if (i+1) % 200 == 0:
        #     print (f'에포크 [{epoch+1}/{num_epochs}], 스텝 [{i+1}/{len(train_loader)}], 손실: {loss.item():.4f}')
```

### 주요 특징:
-   **데이터 로더 (`DataLoader`)**: 대규모 데이터셋을 효율적으로 처리하기 위해 미니 배치 단위로 데이터를 제공합니다.
-   **디바이스 설정 (`.to(device)`)**: 모델과 데이터를 CPU 또는 GPU로 이동하여 훈련 속도를 최적화합니다.
-   **손실 함수 (`nn.CrossEntropyLoss`)**: 다중 클래스 분류에 적합하며, `softmax`와 `NLLLoss`를 결합한 형태입니다.
-   **옵티마이저 (`optim.Adam`)**: 모델의 가중치를 효율적으로 업데이트하는 데 사용됩니다.
-   **훈련 루프**: 각 에포크마다 `DataLoader`를 통해 데이터를 반복하고, 순전파, 손실 계산, 역전파, 파라미터 업데이트를 수행합니다.
-   **`model.train()`**: 모델을 훈련 모드로 설정하여 `Dropout` 등 훈련 시에만 작동하는 레이어를 활성화합니다. (이때 배치 정규화와 드롭아웃은 훈련 모드로 전환되어 정상 작동합니다.)
-   **L2 정규화 (Weight Decay)**: `optim.Adam`의 `weight_decay` 인자를 통해 구현됩니다. 이는 모델의 가중치에 제곱합 패널티를 부여하여 과적합을 방지하고 모델의 일반화 성능을 향상시키는 정규화 기법입니다. 큰 가중치에 대한 패널티를 통해 모델의 복잡도를 제어합니다.

## 3. 모델 추론 (평가)

모델 훈련이 완료된 후에는 훈련되지 않은 데이터(테스트 데이터)에 대해 모델의 성능을 평가하고 예측을 수행합니다.

```python
# 모델 평가 시작
model.eval() # 모델을 평가 모드로 설정합니다. 이는 Dropout 등 훈련 시에만 작동하는 레이어를 비활성화합니다.

# `torch.no_grad()`: 이 컨텍스트 매니저 안에서는 PyTorch가 기울기를 계산하고 저장하는 것을 비활성화합니다.
# 이는 추론 단계에서 메모리 사용량을 줄이고 계산 속도를 높이는 데 도움이 됩니다.
# 평가 단계에서는 파라미터 업데이트가 필요 없으므로 기울기 계산이 불필요합니다.
with torch.no_grad():
    correct = 0
    total = 0
    # 테스트 데이터 로더를 통해 테스트 이미지를 가져옵니다.
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        # `model(images)`: 테스트 이미지를 모델에 입력하여 각 클래스에 대한 로짓(분류 점수)을 얻습니다.
        outputs = model(images)
        # `torch.max(outputs.data, 1)`: 모델 출력(로짓)에서 가장 높은 값을 가진 클래스(인덱스)를 선택합니다.
        # `_`는 최대값을, `predicted`는 해당 값의 인덱스를 반환합니다.
        _, predicted = torch.max(outputs.data, 1)
        # `labels.size(0)`: 현재 배치의 전체 샘플 수입니다.
        total += labels.size(0)
        # `(predicted == labels).sum().item()`: 예측된 클래스와 실제 라벨이 일치하는 샘플의 수를 합산합니다.
        correct += (predicted == labels).sum().item()

    # 정확도 계산: 올바르게 예측한 샘플 수를 전체 샘플 수로 나눕니다.
    accuracy = correct / total
    # print(f'테스트 정확도: {accuracy:.4f}')

# 개별 예측 예시 (선택 사항)
# test_dataset[0]에서 이미지와 라벨을 가져옵니다.
# input_tensor = example_image.unsqueeze(0).to(device)로 배치 차원을 추가하여 모델에 입력합니다.
# output = model(input_tensor)를 통해 예측을 수행하고, torch.max로 최종 예측 클래스를 얻습니다.
```

### 주요 특징:
-   **`model.eval()`**: 모델을 평가 모드로 전환하여, 훈련 중에만 활성화되는 계층(예: `Dropout`, `BatchNorm`)을 비활성화합니다. 이는 평가 시 일관된 결과를 보장합니다. **참고**: `model.eval()` 호출 시 `BatchNorm`은 훈련 시 계산된 평균과 분산을 사용하고, `Dropout`은 완전히 비활성화됩니다.
-   **`torch.no_grad()`**: 이 블록 내에서는 그래디언트 계산을 비활성화하여 메모리 사용량을 줄이고 연산 속도를 높입니다. 추론 시에는 역전파가 필요 없으므로 효율적입니다.
-   **`DataLoader` 사용**: 테스트 데이터셋도 `DataLoader`를 사용하여 효율적으로 배치 단위로 처리합니다.
-   **`torch.max()`**: 모델의 출력(로짓)에서 가장 높은 확률을 가진 클래스를 최종 예측으로 선택하는 데 사용됩니다.
-   **정확도 계산**: 전체 테스트 데이터셋에 대한 모델의 정확도를 계산하여 모델의 일반화 성능을 평가합니다. 