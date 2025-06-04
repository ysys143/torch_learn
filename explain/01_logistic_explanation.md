# `learn/01_logistic.py` 설명

이 문서는 `learn/01_logistic.py` 파일에 구현된 로지스틱 회귀 모델의 핵심 구성 요소인 모델 정의, 훈련 (피팅), 그리고 추론 (평가 및 예측) 과정을 상세히 설명합니다.

## 1. 모델 정의 (`LogisticRegression` 클래스)

로지스틱 회귀 모델은 PyTorch의 `nn.Module`을 상속받아 정의됩니다. 이 모델은 이진 분류 문제에 사용되며, 입력 특성들을 기반으로 0과 1 사이의 확률을 출력합니다.

```python
class LogisticRegression(nn.Module):
    """
    간단한 로지스틱 회귀 모델
    
    로지스틱 회귀는 선형 결합 후 시그모이드 함수를 적용하여
    이진 분류를 수행하는 모델입니다.
    """
    
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        # `nn.Linear`: 입력 특성들을 하나의 출력 값으로 변환하는 선형 레이어입니다.
        # 이 레이어는 `input_size`개의 입력 뉴런과 1개의 출력 뉴런을 가집니다.
        # 즉, `y = xW^T + b` 형태의 선형 변환을 수행합니다.
        self.linear = nn.Linear(input_size, 1)
        # `nn.Sigmoid`: 선형 레이어의 출력을 0과 1 사이의 확률 값으로 변환하는 활성화 함수입니다.
        # 이 함수는 이진 분류 문제에서 클래스에 속할 확률을 나타내기 위해 사용됩니다.
        self.sigmoid = nn.Sigmoid()

        # --- 파라미터 초기화 (Parameter Initialization) ---
        # 신경망의 가중치와 편향을 적절히 초기화하는 것은 학습 안정성과 수렴 속도에 중요합니다.
        # 잘못된 초기화는 기울기 소실(vanishing gradients)이나 폭주(exploding gradients) 문제를 야기할 수 있습니다.
        #
        # PyTorch의 `nn.Linear`는 기본적으로 Kaiming He 초기화(ReLU와 함께 사용 시 적합)나 Xavier/Glorot 초기화(Sigmoid/Tanh와 함께 사용 시 적합)를 사용합니다.
        # 여기서는 명시적으로 Xavier Uniform 초기화를 적용하여 선형 레이어의 가중치를 초기화합니다.
        # 이는 Sigmoid 활성화 함수와 잘 어울립니다. 편향은 0으로 초기화합니다.
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        # --------------------------------------------------
    
    def forward(self, x):
        # `forward` 메서드는 모델이 입력 `x`를 받았을 때 어떻게 계산을 수행할지 정의합니다.
        # 1. `self.linear(x)`: 입력 `x`에 선형 변환을 적용합니다.
        out = self.linear(x)
        # 2. `self.sigmoid(out)`: 선형 변환의 결과에 시그모이드 함수를 적용하여 확률을 계산합니다.
        out = self.sigmoid(out)
        return out
```

### 주요 특징:
-   **단순한 구조**: 단일 선형 레이어와 하나의 시그모이드 활성화 함수로 구성되어 있어 이해하기 쉽습니다.
-   **`nn.Module` 상속**: PyTorch 모델의 기본 클래스로, 모델의 파라미터 관리, `forward` 및 `backward` 메서드 정의 등을 가능하게 합니다.
-   **`nn.Linear`**: 입력 특성들을 가중치와 곱하고 편향을 더하는 선형 변환을 수행합니다.
-   **`nn.Sigmoid`**: 선형 변환의 결과를 이진 분류를 위한 확률로 변환합니다. 출력은 항상 0과 1 사이입니다.
-   **파라미터 초기화 (Xavier Uniform)**: 모델의 가중치와 편향을 학습 시작 전에 특정 값으로 설정하는 과정입니다. `Xavier Uniform` 초기화는 Sigmoid나 Tanh와 같은 활성화 함수와 함께 사용될 때 효과적이며, 학습 초기에 기울기 소실이나 폭주 문제를 완화하여 모델이 안정적으로 학습되도록 돕습니다.

## 2. 모델 훈련 (피팅)

모델 훈련 과정은 정의된 `LogisticRegression` 모델이 주어진 훈련 데이터에 맞춰 최적의 파라미터(가중치와 편향)를 학습하도록 하는 단계입니다.

```python
# 모델 인스턴스 생성
input_size = X_train_tensor.shape[1]
model = LogisticRegression(input_size)

# 손실 함수: 이진 교차 엔트로피 (Binary Cross Entropy)
# `nn.BCELoss()`는 이진 분류 문제에서 모델의 예측 확률과 실제 타겟 값 사이의 차이를 측정합니다.
# 예측이 실제 값에 얼마나 가까운지 나타내며, 이 값을 최소화하는 방향으로 모델이 학습됩니다.
criterion = nn.BCELoss()

# 옵티마이저: Adam
# `optim.Adam`은 모델의 파라미터를 업데이트하여 손실 함수를 최소화하는 알고리즘입니다.
# `model.parameters()`는 훈련 가능한 모델의 모든 파라미터를 옵티마이저에 전달합니다.
# `lr=0.01`은 학습률(learning rate)로, 각 업데이트 단계에서 파라미터가 얼마나 크게 변경될지를 결정합니다.
# `weight_decay` (L2 정규화): 과적합(overfitting)을 방지하는 정규화 기법 중 하나입니다.
# L2 정규화는 모델의 가중치(weights)가 너무 커지는 것을 제어하여 모델의 복잡도를 줄입니다.
# 손실 함수에 가중치 제곱합의 일정 비율을 더함으로써 큰 가중치에 패널티를 부여합니다.
# `weight_decay=0.001`은 이 패널티의 강도를 조절하는 하이퍼파라미터입니다.
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

# 훈련 설정
num_epochs = 1000

# 모델 훈련 (피팅) 시작
model.train()  # 모델을 훈련 모드로 설정합니다. 이는 Dropout이나 Batch Normalization과 같은 특정 레이어의 동작에 영향을 미칩니다.

for epoch in range(num_epochs):
    # 순전파 (Forward pass)
    # 훈련 데이터 `X_train_tensor`를 모델에 입력하여 예측 `outputs`를 얻습니다.
    # `outputs.squeeze()`는 모델 출력의 차원을 조정하여 `y_train_tensor`와 형태를 맞춥니다.
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)
    
    # 역전파 (Backward pass)
    # `optimizer.zero_grad()`: 이전 에포크에서 계산된 기울기를 0으로 초기화합니다.
    # 기울기는 누적되기 때문에 매 에포크마다 초기화해야 올바른 학습이 가능합니다.
    optimizer.zero_grad()
    # `loss.backward()`: 현재 손실에 대한 모델 파라미터들의 기울기를 계산합니다.
    # 이 기울기는 각 파라미터가 손실에 얼마나 기여하는지 나타냅니다.
    loss.backward()
    # `optimizer.step()`: 계산된 기울기를 사용하여 모델의 파라미터(가중치와 편향)를 업데이트합니다.
    # 이 단계에서 모델이 실제로 학습됩니다.
    optimizer.step()
    
    # 손실 기록 (시각화를 위해)
    # `loss.item()`은 손실 텐서에서 Python 숫자를 추출합니다.
    # losses.append(loss.item())
    
    # 진행 상황 출력
    # if (epoch + 1) % 100 == 0:
    #     print(f'에포크 [{epoch+1}/{num_epochs}], 손실: {loss.item():.4f}')
```

### 주요 특징:
-   **손실 함수 (`nn.BCELoss`)**: 이진 분류의 예측 확률과 실제 레이블 간의 차이를 정량화하여 모델이 학습할 목표를 제공합니다.
-   **옵티마이저 (`optim.Adam`)**: 손실 함수를 최소화하기 위해 모델의 가중치를 조정하는 알고리즘입니다. `Adam`은 효과적인 최적화 알고리즘 중 하나입니다.
-   **훈련 루프**: 정해진 에포크 수만큼 반복하며 순전파, 손실 계산, 역전파, 파라미터 업데이트의 과정을 거칩니다.
-   **`model.train()`**: 모델을 훈련 모드로 설정하여 `Dropout` 등 훈련 시에만 작동하는 레이어를 활성화합니다.
-   **기울기 초기화 (`optimizer.zero_grad()`)**: PyTorch는 기본적으로 기울기를 누적하므로, 각 훈련 스텝 전에 기울기를 0으로 초기화하는 것이 중요합니다.
-   **역전파 (`loss.backward()`)**: 계산된 손실을 바탕으로 모델의 각 파라미터에 대한 기울기를 계산합니다.
-   **파라미터 업데이트 (`optimizer.step()`)**: 계산된 기울기를 사용하여 모델의 가중치와 편향을 업데이트합니다.
-   **L2 정규화 (Weight Decay)**: `optim.Adam`의 `weight_decay` 인자를 통해 구현됩니다. 이는 모델의 가중치에 제곱합 패널티를 부여하여 과적합을 방지하고 모델의 일반화 성능을 향상시키는 정규화 기법입니다. 큰 가중치에 대한 패널티를 통해 모델의 복잡도를 제어합니다.

## 3. 모델 추론 (평가 및 예측)

모델 훈련이 완료된 후에는 훈련되지 않은 데이터(테스트 데이터)에 대해 모델의 성능을 평가하고 예측을 수행합니다.

```python
# 모델 평가 시작
model.eval()  # 모델을 평가 모드로 설정합니다. 이는 Dropout 등 훈련 시에만 작동하는 레이어를 비활성화합니다.

# `torch.no_grad()`: 이 컨텍스트 매니저 안에서는 PyTorch가 기울기를 계산하고 저장하는 것을 비활성화합니다.
# 이는 추론 단계에서 메모리 사용량을 줄이고 계산 속도를 높이는 데 도움이 됩니다.
# 평가 단계에서는 파라미터 업데이트가 필요 없으므로 기울기 계산이 불필요합니다.
with torch.no_grad():
    # 훈련 데이터 예측
    # 훈련 데이터 `X_train_tensor`에 대해 모델의 예측 확률을 얻습니다.
    train_outputs = model(X_train_tensor)
    # 예측 확률(`train_outputs.squeeze()`)을 0.5를 기준으로 이진 분류 결과로 변환합니다.
    # 0.5보다 크면 1(생존), 아니면 0(사망)으로 간주합니다. `.float()`으로 텐서 타입을 맞춥니다.
    train_predictions = (train_outputs.squeeze() > 0.5).float()
    # `accuracy_score`를 사용하여 실제 훈련 타겟(`y_train_tensor`)과 예측 결과(`train_predictions`)를 비교하여 정확도를 계산합니다.
    train_accuracy = accuracy_score(y_train_tensor, train_predictions)
    
    # 테스트 데이터 예측
    # 훈련 과정에서 사용되지 않은 테스트 데이터 `X_test_tensor`에 대해 예측을 수행합니다.
    # 이 정확도가 모델의 일반화 성능을 더 잘 나타냅니다.
    test_outputs = model(X_test_tensor)
    test_predictions = (test_outputs.squeeze() > 0.5).float()
    test_accuracy = accuracy_score(y_test_tensor, test_predictions)

# 정확도 출력
# print(f"훈련 정확도: {train_accuracy:.4f}")
# print(f"테스트 정확도: {test_accuracy:.4f}")
```

### 주요 특징:
-   **`model.eval()`**: 모델을 평가 모드로 전환하여, 훈련 중에만 활성화되는 계층(예: `Dropout`, `BatchNorm`)을 비활성화합니다. 이는 평가 시 일관된 결과를 보장합니다.
-   **`torch.no_grad()`**: 이 블록 내에서는 그래디언트 계산을 비활성화하여 메모리 사용량을 줄이고 연산 속도를 높입니다. 추론 시에는 역전파가 필요 없으므로 효율적입니다.
-   **임계값 기반 예측**: 모델의 출력(확률)을 0.5와 같은 임계값과 비교하여 최종 이진 예측(0 또는 1)을 결정합니다.
-   **정확도 평가**: `sklearn.metrics.accuracy_score`를 사용하여 실제 레이블과 모델의 예측 결과를 비교하여 정확도를 계산합니다. 테스트 데이터에 대한 정확도는 모델의 일반화 성능을 나타냅니다. 