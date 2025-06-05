# `learn/03_autoencoder.py` 설명

이 문서는 `learn/03_autoencoder.py` 파일에 구현된 오토인코더 모델의 핵심 구성 요소인 모델 정의, 훈련 (피팅), 그리고 추론 (재구성) 과정을 상세히 설명합니다.

## 1. 모델 정의 (`Autoencoder` 클래스)

오토인코더 모델은 PyTorch의 `nn.Module`을 상속받아 정의됩니다. 이 모델은 입력 데이터를 압축하여 잠재 공간 표현으로 만들고(인코딩), 이 잠재 표현을 다시 원본 데이터 형태로 복원(디코딩)하는 신경망입니다. 입력과 출력이 같아지는 것을 목표로 하며, 주로 차원 축소나 특징 학습과 같은 비지도 학습에 활용됩니다.

```python
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
        # 인코더는 원본 이미지를 받아 더 낮은 차원의 잠재 표현으로 압축하는 역할을 합니다.
        # `nn.Sequential`을 사용하여 여러 레이어를 순차적으로 연결합니다.
        # - `nn.Linear(input_size, hidden_size)`: 입력 이미지(784픽셀)를 `hidden_size` 뉴런으로 변환합니다.
        # - `nn.BatchNorm1d(hidden_size)`: 배치 정규화. 각 미니 배치마다 입력의 평균과 분산을 정규화하여 학습 안정성을 높이고 수렴 속도를 향상시킵니다.
        # - `nn.ReLU()`: 비선형성을 추가합니다.
        # - `nn.Dropout(p=0.2)`: 드롭아웃. 훈련 시 무작위로 일부 뉴런을 비활성화하여 과적합을 방지합니다.
        # - `nn.Linear(hidden_size, latent_dim)`: 중간층을 `latent_dim` 크기의 잠재 공간으로 압축합니다.
        # - `nn.BatchNorm1d(latent_dim)`: 배치 정규화.
        # - `nn.ReLU()`: 잠재 공간에도 비선형성을 적용하여 표현력을 높입니다.
        # - `nn.Dropout(p=0.2)`: 드롭아웃.
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
        # 디코더 (Decoder): 잠재 공간 -> 출력
        # 디코더는 인코딩된 잠재 표현을 받아 원본 이미지와 유사한 형태로 복원하는 역할을 합니다.
        # - `nn.Linear(latent_dim, hidden_size)`: 잠재 공간에서 `hidden_size` 뉴런으로 확장합니다.
        # - `nn.BatchNorm1d(hidden_size)`: 배치 정규화.
        # - `nn.ReLU()`: 비선형성을 추가합니다.
        # - `nn.Dropout(p=0.2)`: 드롭아웃.
        # - `nn.Linear(hidden_size, input_size)`: 중간층을 원본 이미지 크기(`input_size`)로 복원합니다.
        # - `nn.Tanh()`: 디코더의 최종 출력 활성화 함수입니다. 데이터 정규화가 -1에서 1 사이로 이루어졌기 때문에 `Tanh`를 사용하여 출력 범위가 입력 데이터의 범위와 일치하도록 합니다. 이는 재구성 품질에 중요합니다.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )

        # --- 파라미터 초기화 (Parameter Initialization) ---
        # 신경망의 가중치와 편향을 적절히 초기화하는 것은 학습 안정성과 수렴 속도에 중요합니다.
        # ReLU 활성화 함수를 사용하는 레이어의 가중치는 Kaiming He 초기화가 적합합니다.
        # Tanh 활성화 함수를 사용하는 레이어의 가중치는 Xavier/Glorot 초기화가 적합합니다.
        
        # 인코더 레이어 초기화
        # 첫 번째 Linear 레이어 (input_size -> hidden_size)
        nn.init.kaiming_uniform_(self.encoder[0].weight, nonlinearity='relu')
        if self.encoder[0].bias is not None:
            nn.init.zeros_(self.encoder[0].bias)

        # 두 번째 Linear 레이어 (hidden_size -> latent_dim)
        nn.init.kaiming_uniform_(self.encoder[4].weight, nonlinearity='relu')
        if self.encoder[4].bias is not None:
            nn.init.zeros_(self.encoder[4].bias)

        # 디코더 레이어 초기화
        # 첫 번째 Linear 레이어 (latent_dim -> hidden_size)
        nn.init.kaiming_uniform_(self.decoder[0].weight, nonlinearity='relu')
        if self.decoder[0].bias is not None:
            nn.init.zeros_(self.decoder[0].bias)
            
        # Tanh를 사용하는 마지막 Linear 레이어 (hidden_size -> input_size)는 Xavier 초기화가 더 적합할 수 있습니다.
        nn.init.xavier_uniform_(self.decoder[4].weight)
        if self.decoder[4].bias is not None:
            nn.init.zeros_(self.decoder[4].bias)
        # --------------------------------------------------

    def forward(self, x):
        # `forward` 메서드는 모델이 입력 `x`를 받았을 때 어떻게 계산을 수행할지 정의합니다.
        # 1. `x.view(x.size(0), -1)`: 입력 이미지는 (배치 크기, 채널, 높이, 너비) 형태입니다.
        #    이를 (배치 크기, 픽셀 수) 형태의 1차원 벡터로 평탄화합니다. (예: (256, 1, 28, 28) -> (256, 784)).
        x = x.view(x.size(0), -1)
        
        # 2. `self.encoder(x)`: 평탄화된 입력 `x`를 인코더에 통과시켜 잠재 표현 `encoded`를 얻습니다.
        encoded = self.encoder(x)
        
        # 3. `self.decoder(encoded)`: 잠재 표현 `encoded`를 디코더에 통과시켜 재구성된 이미지 `decoded`를 얻습니다.
        decoded = self.decoder(encoded)
        
        return decoded
```

### 주요 특징:
-   **인코더-디코더 구조**: 데이터를 압축하고 다시 복원하는 두 부분으로 구성되어 있습니다.
-   **`nn.Sequential`**: 여러 레이어를 하나의 블록으로 묶어 모델을 간결하게 정의할 수 있도록 합니다.
-   **`nn.Linear`**: 완전 연결층으로, 데이터의 차원을 변경하는 데 사용됩니다.
-   **`nn.ReLU` 활성화 함수**: 인코더와 디코더의 중간 레이어에서 비선형성을 부여하여 모델의 표현력을 높입니다.
-   **`nn.BatchNorm1d` (배치 정규화)**: 각 레이어의 입력을 정규화하여 내부 공변량 변화를 줄이고 학습을 안정화하며 수렴 속도를 높입니다. 이는 더 높은 학습률을 사용할 수 있게 돕고 모델의 일반화 성능을 향상시킵니다.
-   **`nn.Dropout` (드롭아웃)**: 훈련 시 무작위로 뉴런의 일부를 비활성화하여 모델이 특정 특성이나 뉴런에 과도하게 의존하는 것을 방지합니다. 이는 앙상블 효과를 주어 과적합을 줄이고 모델의 일반화 성능을 높이는 데 효과적입니다.
-   **`nn.Tanh` 활성화 함수 (디코더 출력)**: 입력 데이터가 -1과 1 사이로 정규화되었으므로, 디코더의 마지막 레이어에서 `Tanh`를 사용하여 출력도 동일한 범위로 재구성되도록 합니다. 이는 재구성된 이미지의 품질에 결정적인 영향을 미칩니다.
-   **이미지 평탄화**: `forward` 메서드에서 `x.view()`를 사용하여 다차원 이미지 데이터를 1차원 벡터로 변환하여 완전 연결층의 입력 요구사항에 맞춥니다.
-   **`nn.Module` 상속**: PyTorch 모델의 표준이며, 파라미터 관리 및 `forward`/`backward` 메서드 정의를 용이하게 합니다.
-   **파라미터 초기화 (Kaiming He, Xavier Uniform)**: 모델의 가중치와 편향을 학습 시작 전에 특정 값으로 설정하는 과정입니다. ReLU 활성화 함수를 사용하는 레이어에는 `Kaiming He` 초기화가, Tanh 활성화 함수를 사용하는 레이어에는 `Xavier Uniform` 초기화가 적합하며, 이는 학습 초기에 기울기 소실이나 폭주 문제를 완화하여 모델이 안정적으로 학습되도록 돕습니다.

## 2. 모델 훈련 (피팅)

오토인코더 모델 훈련 과정은 정의된 `Autoencoder` 모델이 MNIST 훈련 데이터셋에 대해 원본 이미지를 재구성하는 능력을 학습하도록 하는 단계입니다.

```python
# 모델 인스턴스 생성
input_size = 28 * 28  # MNIST 이미지 크기 (784)
hidden_size = 128     # 인코더/디코더의 중간층 뉴런 수
latent_dim = 32       # 잠재 공간의 차원 (원본보다 작게 설정)

model = Autoencoder(input_size, hidden_size, latent_dim)

# GPU 사용 가능 시 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 손실 함수: MSELoss (Mean Squared Error)
# `nn.MSELoss()`는 재구성 작업에 일반적으로 사용되는 손실 함수입니다.
# 이는 모델의 출력(재구성된 이미지)과 실제 입력(원본 이미지) 사이의 픽셀 값 차이의 제곱 평균을 계산합니다.
# 이 손실을 최소화함으로써 모델은 원본 이미지를 최대한 정확하게 재구성하도록 학습됩니다.
criterion = nn.MSELoss()

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
num_epochs = 20

# 모델 훈련 (피팅) 시작
model.train() # 모델을 훈련 모드로 설정합니다. 이는 Dropout이나 Batch Normalization과 같은 특정 레이어의 동작에 영향을 미칩니다.

for epoch in range(num_epochs):
    # `train_loader`를 통해 미니 배치 단위로 훈련 이미지를 가져옵니다.
    # 오토인코더는 비지도 학습이므로, 입력 이미지가 곧 타겟이 됩니다. 따라서 `labels`는 사용되지 않습니다 (`_`).
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        
        # 순전파
        # `model(images)`: 훈련 이미지를 모델에 입력하여 재구성된 출력 `outputs`를 얻습니다.
        # `criterion(outputs, images.view(images.size(0), -1))`: 재구성된 출력과 원본 이미지(평탄화된 형태)를 비교하여 손실을 계산합니다.
        outputs = model(images)
        loss = criterion(outputs, images.view(images.size(0), -1)) # 출력과 원본 이미지 비교
        
        # 역전파 및 최적화
        # `optimizer.zero_grad()`: 이전 스텝에서 계산된 기울기를 0으로 초기화합니다.
        # `loss.backward()`: 현재 손실에 대한 모델 파라미터들의 기울기를 계산합니다.
        # `optimizer.step()`: 계산된 기울기를 사용하여 모델의 파라미터를 업데이트합니다.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 진행 상황 출력 (선택 사항)
        # if (i+1) % 100 == 0:
        #     print (f'에포크 [{epoch+1}/{num_epochs}], 스텝 [{i+1}/{len(train_loader)}], 손실: {loss.item():.4f}')
    
    # 손실 기록 (시각화를 위해)
    # losses.append(loss.item())
```

### 주요 특징:
-   **손실 함수 (`nn.MSELoss`)**: 재구성 모델의 성능을 측정하는 데 사용되며, 예측값과 실제값 사이의 제곱 오차를 최소화합니다.
-   **옵티마이저 (`optim.Adam`)**: 모델의 가중치를 효율적으로 업데이트하여 손실을 줄입니다.
-   **비지도 학습**: 입력 이미지가 그대로 모델의 타겟이 되므로, 별도의 라벨 데이터가 필요 없습니다.
-   **훈련 루프**: 각 에포크마다 `DataLoader`를 통해 데이터를 반복하고, 순전파, 손실 계산, 역전파, 파라미터 업데이트를 수행합니다.
-   **`model.train()`**: 모델을 훈련 모드로 설정하여 `Dropout`이나 `BatchNorm`과 같은 특정 레이어가 훈련 시에만 작동하도록 합니다.
-   **L2 정규화 (Weight Decay)**: `optim.Adam`의 `weight_decay` 인자를 통해 구현됩니다. 이는 모델의 가중치에 제곱합 패널티를 부여하여 과적합을 방지하고 모델의 일반화 성능을 향상시키는 정규화 기법입니다. 큰 가중치에 대한 패널티를 통해 모델의 복잡도를 제어합니다.

## 3. 모델 추론 (재구성 및 시각화)

모델 훈련이 완료된 후에는 훈련되지 않은 데이터(테스트 데이터)에 대해 모델의 재구성 성능을 평가하고, 원본 이미지와 재구성된 이미지를 시각화하여 모델의 학습 결과를 확인합니다.

```python
# 모델 평가 시작
model.eval() # 모델을 평가 모드로 설정합니다. 이는 Dropout 등 훈련 시에만 작동하는 레이어를 비활성화합니다.

# `torch.no_grad()`: 이 컨텍스트 매니저 안에서는 PyTorch가 기울기를 계산하고 저장하는 것을 비활성화합니다.
# 이는 추론 단계에서 메모리 사용량을 줄이고 계산 속도를 높이는 데 도움이 됩니다.
# 평가 단계에서는 파라미터 업데이트가 필요 없으므로 기울기 계산이 불필요합니다.
with torch.no_grad():
    # 테스트 데이터셋에서 몇 개 샘플 가져오기
    # `test_loader`에서 첫 번째 배치 이미지와 라벨을 가져옵니다.
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    
    # 원본 이미지를 오토인코더에 통과시켜 재구성
    # `model(images)`: 테스트 이미지를 모델에 입력하여 재구성된 이미지를 얻습니다.
    # `.cpu()`: 결과를 GPU에서 CPU로 이동하여 Matplotlib으로 시각화할 수 있도록 합니다.
    reconstructed_images = model(images).cpu()
    
    # 이미지 정규화 해제 및 시각화 준비
    # 원본 이미지와 재구성된 이미지는 픽셀 값이 -1에서 1 사이로 정규화되어 있습니다.
    # 시각화를 위해 0에서 1 사이로 다시 변환합니다 (`/ 2 + 0.5`).
    # `view()`를 사용하여 이미지를 다시 (배치 크기, 채널, 높이, 너비) 형태로 만듭니다.
    original_images_display = images.cpu().view(images.size(0), 1, 28, 28) / 2 + 0.5
    reconstructed_images_display = reconstructed_images.view(images.size(0), 1, 28, 28) / 2 + 0.5
    
    # 첫 10개 이미지 시각화
    # `matplotlib.pyplot`을 사용하여 원본 이미지와 재구성된 이미지를 나란히 표시합니다.
    # `fig, axes`는 그림(figure)과 서브플롯(axes) 배열을 생성합니다.
    # `zip([original_images_display, reconstructed_images_display], axes)`로 원본/재구성 이미지와 서브플롯 행을 묶어 처리합니다.
    # `np.squeeze(img.numpy())`는 PyTorch 텐서를 NumPy 배열로 변환하고 단일 차원(채널)을 제거합니다.
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))
    for images, row in zip([original_images_display, reconstructed_images_display], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img.numpy()), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
    # 재구성 이미지 저장
    plt.savefig('autoencoder_reconstruction.png', dpi=150, bbox_inches='tight')
    # print("원본 및 재구성된 이미지 결과가 'autoencoder_reconstruction.png'로 저장되었습니다.\n")

# 손실 그래프 시각화 (선택 사항)
# plt.figure(figsize=(8, 4))
# plt.plot(losses)
# plt.title('오토인코더 훈련 손실 (Autoencoder Training Loss)')
# plt.xlabel('에포크 (Epoch)')
# plt.ylabel('손실 (Loss)')
# plt.grid(True)
# plt.savefig('autoencoder_loss.png', dpi=150, bbox_inches='tight')
# print("손실 그래프가 'autoencoder_loss.png'로 저장되었습니다.\n")
```

### 주요 특징:
-   **`model.eval()`**: 모델을 평가 모드로 전환하여, 훈련 중에만 활성화되는 계층(예: `Dropout`, `BatchNorm`)을 비활성화합니다. 이는 평가 시 일관된 결과를 보장합니다. **참고**: `model.eval()` 호출 시 `BatchNorm`은 훈련 시 계산된 평균과 분산을 사용하고, `Dropout`은 완전히 비활성화됩니다.
-   **`torch.no_grad()`**: 이 블록 내에서는 그래디언트 계산을 비활성화하여 메모리 사용량을 줄이고 연산 속도를 높입니다. 추론 시에는 역전파가 필요 없으므로 효율적입니다.
-   **이미지 재구성**: 훈련된 오토인코더에 테스트 이미지를 입력하여 재구성된 이미지를 얻습니다.
-   **정규화 해제**: 시각화를 위해 정규화된 픽셀 값을 다시 0-1 또는 0-255 범위로 변환합니다.
-   **시각화 (`matplotlib.pyplot`)**: 원본 이미지와 재구성된 이미지를 나란히 표시하여 모델의 재구성 품질을 직관적으로 확인할 수 있도록 합니다.
-   **결과 저장**: 재구성된 이미지와 손실 그래프를 파일로 저장하여 훈련 결과를 영구적으로 확인할 수 있도록 합니다.
