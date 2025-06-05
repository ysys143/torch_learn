# 0. PyTorch 기본 구성 요소 (Core Components) - 설명

이 문서는 PyTorch의 가장 기본적인 구성 요소들에 대한 상세한 설명을 제공합니다. `learn/00_pytorch_core_components.py` 파일의 코드 예제와 함께 학습하면 더욱 효과적입니다.

## 목차

1.  [텐서 (Tensors)](#1-텐서-tensors)
2.  [자동 미분 (`torch.autograd`)](#2-자동-미분-torchautograd)
3.  [모듈 및 클래스 기본 (`torch.nn.Module` 소개)](#3-모듈-및-클래스-기본-torchnnmodule-소개)
4.  [활성화 함수 (Activation Functions)](#4-활성화-함수-activation-functions)
5.  [손실 함수 (Loss Functions)](#5-손실-함수-loss-functions)
6.  [옵티마이저 (`torch.optim`)](#6-옵티마이저-torchoptim)
7.  [장치 할당 (`.to(device)`)](#7-장치-할당-todevice)
8.  [가중치 초기화 (Weight Initialization)](#8-가중치-초기화-weight-initialization)

---

## 1. 텐서 (Tensors)

PyTorch에서 텐서(`torch.Tensor`)는 핵심적인 데이터 구조입니다. 다차원 배열의 형태로 데이터를 표현하며, NumPy의 `ndarray`와 유사하지만 GPU를 사용한 연산 가속 기능을 제공한다는 큰 차이점이 있습니다.

### 1.1. 텐서 생성

다양한 방법으로 텐서를 생성할 수 있습니다:

*   **데이터로부터 직접 생성**: Python 리스트나 시퀀스 데이터를 `torch.tensor()`를 사용하여 텐서로 변환합니다.
    ```python
    data = [[1, 2],[3, 4]]
    x_data = torch.tensor(data)
    ```
*   **NumPy 배열로부터 생성**: NumPy 배열을 `torch.from_numpy()`를 사용하여 텐서로 변환할 수 있습니다. (메모리 공유 주의: NumPy 배열 변경 시 텐서도 변경됨)
    ```python
    # import numpy as np
    # np_array = np.array(data)
    # x_np = torch.from_numpy(np_array)
    ```
*   **다른 텐서로부터 생성**:
    *   `torch.ones_like(old_tensor)`: 기존 텐서의 모양, 자료형, 장치를 유지하면서 모든 요소가 1인 텐서를 생성합니다.
    *   `torch.zeros_like(old_tensor)`: 기존 텐서의 모양, 자료형, 장치를 유지하면서 모든 요소가 0인 텐서를 생성합니다.
    *   `torch.rand_like(old_tensor, dtype=torch.float)`: 기존 텐서의 모양과 장치를 유지하면서 0과 1 사이의 균등 분포로 랜덤 값을 채웁니다. `dtype`을 명시하여 자료형을 변경할 수 있습니다.
*   **특정 모양(shape)으로 생성**:
    *   `torch.rand(shape)`: 주어진 모양으로 0과 1 사이 균등 분포 랜덤 값을 갖는 텐서를 생성합니다.
    *   `torch.randn(shape)`: 주어진 모양으로 평균 0, 표준편차 1의 정규 분포 랜덤 값을 갖는 텐서를 생성합니다.
    *   `torch.ones(shape)`: 주어진 모양으로 모든 요소가 1인 텐서를 생성합니다.
    *   `torch.zeros(shape)`: 주어진 모양으로 모든 요소가 0인 텐서를 생성합니다.
    *   `torch.empty(shape)`: 주어진 모양으로 초기화되지 않은 (메모리에 남아있는 임의의 값) 텐서를 생성합니다.
    *   `torch.arange(start, end, step)`: 주어진 범위 내에서 일정한 간격의 값을 갖는 1차원 텐서를 생성합니다.
    *   `torch.linspace(start, end, steps)`: 주어진 범위 내에서 균등한 간격의 `steps` 개 값을 갖는 1차원 텐서를 생성합니다.

### 1.2. 텐서의 속성 (Attributes)

텐서는 다음과 같은 주요 속성을 가집니다:

*   `tensor.shape` 또는 `tensor.size()`: 텐서의 각 차원의 크기를 나타내는 튜플입니다. (예: `(2, 3)`)
*   `tensor.dtype`: 텐서에 저장된 데이터의 자료형입니다. (예: `torch.float32`, `torch.int64`)
    *   자주 사용되는 자료형: `torch.float32` (또는 `torch.float`), `torch.float64` (또는 `torch.double`), `torch.float16` (또는 `torch.half`), `torch.int8`, `torch.int16`, `torch.int32`, `torch.int64` (또는 `torch.long`), `torch.bool`.
*   `tensor.device`: 텐서가 저장된 장치 (CPU 또는 GPU)를 나타냅니다. (예: `cpu`, `cuda:0`)
*   `tensor.ndim` 또는 `tensor.dim()`: 텐서의 차원 수 (축의 개수).
*   `tensor.requires_grad`: 이 텐서에 대해 기울기를 계산할지 여부를 나타내는 불리언 값. 자동 미분 섹션에서 자세히 다룹니다.

### 1.3. 텐서 연산

PyTorch는 광범위한 텐서 연산을 지원합니다.

*   **인덱싱 및 슬라이싱**: NumPy와 유사한 방식으로 텐서의 특정 요소나 부분에 접근할 수 있습니다.
    ```python
    tensor = torch.ones(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}") # ...은 필요한 만큼의 :를 의미
    tensor[:,1] = 0 # 특정 열의 모든 값을 0으로 변경
    ```
*   **텐서 합치기**:
    *   `torch.cat(tensors, dim=0)`: 주어진 차원(`dim`)을 따라 텐서들을 이어 붙입니다. 다른 차원의 크기는 동일해야 합니다.
    *   `torch.stack(tensors, dim=0)`: 주어진 차원(`dim`)에 새로운 차원을 추가하여 텐서들을 쌓습니다. 모든 텐서는 동일한 모양을 가져야 합니다.
*   **산술 연산**:
    *   **요소별(element-wise) 연산**: `+`, `-`, `*`, `/`, `**` (거듭제곱) 등.
        ```python
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        c = a + b # tensor([5, 7, 9])
        d = a * b # tensor([4, 10, 18])
        ```
    *   **행렬 곱(matrix multiplication)**:
        *   `tensor1 @ tensor2` (Python 3.5+ 연산자)
        *   `torch.matmul(tensor1, tensor2)`
        *   `tensor1.matmul(tensor2)`
    *   **브로드캐스팅(Broadcasting)**: NumPy와 유사하게, 특정 조건 하에서 모양이 다른 텐서 간의 연산을 가능하게 합니다.
*   **단일 요소 텐서 변환**:
    *   `tensor.sum()`, `tensor.mean()`, `tensor.max()`, `tensor.min()` 등은 텐서 전체 또는 특정 차원에 대한 집계 연산을 수행합니다.
    *   결과가 단일 요소 텐서일 경우, `.item()` 메소드를 사용하여 Python 숫자로 변환할 수 있습니다.
        ```python
        agg = tensor.sum()
        agg_item = agg.item()
        ```
*   **바꿔치기(In-place) 연산**: 연산 결과를 피연산자 자체에 저장하는 연산입니다. 함수명 뒤에 밑줄(`_`)이 붙습니다. (예: `tensor.add_(5)`, `tensor.copy_(other_tensor)`, `tensor.t_()`) 메모리 효율성을 높일 수 있지만, 기울기 계산 시 문제를 일으킬 수 있으므로 주의해야 합니다. (파생된 텐서가 아닌 리프(leaf) 텐서에만 사용 권장)

### 1.4. NumPy 변환 (Bridge)

*   `tensor.numpy()`: PyTorch 텐서를 NumPy 배열로 변환합니다. CPU 상의 텐서만 변환 가능하며, 변환된 배열과 원래 텐서는 메모리를 공유합니다. (하나를 변경하면 다른 하나도 변경됨)
*   `torch.from_numpy(numpy_array)`: NumPy 배열을 PyTorch 텐서로 변환합니다. 마찬가지로 메모리를 공유합니다.

---

## 2. 자동 미분 (`torch.autograd`)

`torch.autograd`는 PyTorch의 자동 미분 엔진으로, 신경망 학습의 핵심인 역전파(backpropagation) 알고리즘을 매우 쉽게 구현할 수 있도록 지원합니다.

### 2.1. `requires_grad`

텐서를 생성할 때 `requires_grad=True`로 설정하면, 해당 텐서에 대한 모든 연산들이 추적됩니다. 연산이 완료된 후 `.backward()`를 호출하면 모든 기울기가 자동으로 계산됩니다. 이 텐서에 대한 기울기는 `.grad` 속성에 누적됩니다.

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

print(f"x: {x}") # x는 사용자가 직접 생성했으므로 leaf variable
print(f"y: {y}") # y는 연산을 통해 생성, y.grad_fn은 <AddBackward0>
print(f"z: {z}") # z는 연산을 통해 생성, z.grad_fn은 <MulBackward0>
print(f"out: {out}") # out은 연산을 통해 생성, out.grad_fn은 <MeanBackward0>

out.backward() # out에 대한 기울기 계산 (dout/dx)
print(x.grad) # x.grad에는 d(out)/dx가 저장됨
```

*   **`grad_fn`**: `requires_grad=True`인 텐서에 연산이 수행되면, 결과 텐서는 연산을 나타내는 `grad_fn` (gradient function) 객체를 갖게 됩니다. 이 객체는 역전파 시 기울기를 계산하는 데 사용됩니다. 사용자가 직접 생성한 텐서(leaf tensor)는 `grad_fn`이 `None`입니다.
*   **`.backward(gradient=None, retain_graph=None, create_graph=False)`**:
    *   스칼라 값(예: 손실 함수 결과)에 대해 호출하면 자동으로 `gradient=torch.tensor(1.0)`으로 간주하고 기울기를 계산합니다.
    *   텐서에 대해 호출하려면, 해당 텐서와 동일한 모양의 `gradient` 인자를 전달해야 합니다 (벡터-야코비안 곱).
    *   `retain_graph=True`: 여러 번 `.backward()`를 호출해야 할 때 계산 그래프를 유지합니다. 기본값은 `False`로, 한 번 `.backward()` 후에는 그래프가 해제되어 메모리를 절약합니다.
    *   `create_graph=True`: 고계도 미분(higher-order derivatives)을 계산할 수 있도록 미분 그래프를 생성합니다.
*   **기울기 누적**: 기울기는 `.grad` 속성에 누적(accumulate)됩니다. 따라서 각 학습 단계(iteration) 전에 기울기를 0으로 초기화해야 합니다. (`optimizer.zero_grad()` 또는 수동으로 `tensor.grad.zero_()`)

### 2.2. 기울기 추적 멈추기

특정 상황에서는 기울기 계산이 필요하지 않거나, 모델의 일부 파라미터를 고정(freeze)하고 싶을 수 있습니다.

*   **`torch.no_grad()` 컨텍스트 매니저**: 이 블록 내의 연산들은 기울기를 추적하지 않습니다. 주로 모델 평가(evaluation)나 추론(inference) 시에 사용하여 메모리 사용량을 줄이고 계산 속도를 높입니다.
    ```python
    with torch.no_grad():
        # 이 블록 안의 연산은 기울기를 계산하지 않음
        y = x * 2
    print(y.requires_grad) # False
    ```
*   **`.detach()` 메소드**: 현재 계산 기록으로부터 분리된 새로운 텐서를 반환합니다. 이 텐서는 원본 텐서와 데이터를 공유하지만, 기울기 계산 기록은 갖지 않습니다. 즉, `requires_grad`가 `False`가 됩니다.
    ```python
    x = torch.randn(3, requires_grad=True)
    y = x.detach()
    print(y.requires_grad) # False
    print(x.eq(y).all())   # True (데이터는 동일)
    ```
*   **`tensor.requires_grad_(False)` (In-place)**: 텐서 자체의 `requires_grad` 속성을 직접 변경합니다.

---

## 3. 모듈 및 클래스 기본 (`torch.nn.Module` 소개)

`torch.nn` 네임스페이스는 신경망을 구성하기 위한 다양한 모듈과 클래스를 제공합니다. 모든 신경망 모듈의 기본 클래스는 `torch.nn.Module`입니다.

*   **`nn.Module` 상속**: 사용자 정의 모델이나 레이어를 만들려면 `nn.Module`을 상속받아야 합니다.
*   **`__init__(self)` 메소드**: 모델의 구성 요소(레이어, 다른 모듈 등)를 정의합니다. `super().__init__()`를 먼저 호출해야 합니다.
*   **`forward(self, input)` 메소드**: 입력 데이터를 받아 출력 데이터를 반환하는 순전파(forward pass) 연산을 정의합니다. 이 메소드 내에서 `__init__`에서 정의한 모듈들을 사용합니다.
*   **파라미터 자동 등록**: `nn.Module`의 속성으로 `nn.Parameter` 객체나 다른 `nn.Module` 객체를 할당하면 자동으로 모델의 파라미터로 등록되어, `model.parameters()` 등을 통해 접근할 수 있고 옵티마이저에 전달될 수 있습니다.
    *   `nn.Parameter`: `torch.Tensor`의 특별한 하위 클래스로, `nn.Module`에 속성으로 할당될 때 자동으로 파라미터로 등록됩니다. 기본적으로 `requires_grad=True`입니다.
*   **주요 `nn` 모듈**:
    *   `nn.Linear(in_features, out_features)`: 완전 연결 계층(fully connected layer).
    *   `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`: 2D 컨볼루션 계층.
    *   `nn.ReLU()`, `nn.Sigmoid()`, `nn.Tanh()`: 활성화 함수.
    *   `nn.CrossEntropyLoss()`, `nn.MSELoss()`: 손실 함수.
    *   `nn.BatchNorm2d(num_features)`: 배치 정규화(Batch Normalization).
    *   `nn.Dropout(p=0.5)`: 드롭아웃.
    *   `nn.Embedding(num_embeddings, embedding_dim)`: 임베딩 층.
    *   `nn.LSTM`, `nn.GRU`: 순환 신경망(RNN) 계층.
    *   `nn.Sequential`: 여러 모듈을 순차적으로 담는 컨테이너.

```python
import torch.nn as nn
import torch.nn.functional as F # functional API (주로 상태가 없는 연산)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5) # 입력 채널 1, 출력 채널 20, 커널 크기 5x5
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4*4*50, 500) # 입력 특성 수, 출력 특성 수
        self.fc2 = nn.Linear(500, 10)     # 클래스 개수 10개로 가정

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2) # 2x2 윈도우, 스트라이드 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)    # Flatten: 배치 크기는 유지(-1), 나머지는 일렬로
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) # 분류 문제용 출력 (NLLLoss와 함께 사용)

model_example = SimpleModel()
print(model_example)
# model_example.parameters()를 통해 모든 학습 가능한 파라미터에 접근 가능
```
`torch.nn.functional` (보통 `F`로 import)은 활성화 함수, 풀링, 손실 함수 등 상태(학습 가능한 파라미터)가 없는 연산들을 함수 형태로 제공합니다. `nn.Module` 형태로 제공되는 것들은 내부에 파라미터를 가질 수 있습니다.

---

## 4. 활성화 함수 (Activation Functions)

활성화 함수는 신경망에 비선형성(non-linearity)을 도입하여 모델이 복잡한 패턴을 학습할 수 있도록 합니다. `torch.nn` 모듈 또는 `torch.nn.functional`을 통해 사용할 수 있습니다.

*   **`nn.ReLU()` / `F.relu(input)` (Rectified Linear Unit)**:
    *   `f(x) = max(0, x)`
    *   가장 널리 사용되는 활성화 함수 중 하나입니다. 계산이 간단하고, 특정 조건 하에서 기울기 소실(vanishing gradient) 문제를 완화합니다.
    *   단점: 입력이 음수이면 기울기가 0이 되어 해당 뉴런이 학습 과정에서 "죽는(dying ReLU)" 현상이 발생할 수 있습니다.
*   **`nn.Sigmoid()` / `torch.sigmoid(input)`**:
    *   `f(x) = 1 / (1 + exp(-x))`
    *   출력 값을 0과 1 사이로 압축합니다. 주로 이진 분류 문제의 출력층이나 특정 확률 값을 나타낼 때 사용됩니다.
    *   단점: 입력 값이 매우 크거나 작으면 기울기가 0에 가까워져 기울기 소실 문제가 발생하기 쉽습니다. 출력의 중심이 0.5여서 학습을 느리게 할 수 있습니다.
*   **`nn.Tanh()` / `torch.tanh(input)` (Hyperbolic Tangent)**:
    *   `f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
    *   출력 값을 -1과 1 사이로 압축합니다. Sigmoid보다 출력의 중심이 0이어서 학습 효율이 더 좋을 수 있습니다.
    *   단점: Sigmoid와 마찬가지로 기울기 소실 문제가 발생할 수 있습니다.
*   **`nn.LeakyReLU(negative_slope=0.01)` / `F.leaky_relu(input, negative_slope=0.01)`**:
    *   `f(x) = x if x > 0 else negative_slope * x`
    *   ReLU의 "죽는 뉴런" 문제를 해결하기 위해 입력이 음수일 때 작은 기울기(`negative_slope`)를 갖도록 합니다.
*   **`nn.Softmax(dim=None)` / `F.softmax(input, dim=None)`**:
    *   입력 벡터의 각 요소를 0과 1 사이의 값으로 변환하고, 모든 요소의 합이 1이 되도록 정규화합니다. 주로 다중 클래스 분류 문제의 출력층에서 각 클래스에 대한 확률 분포를 얻기 위해 사용됩니다.
    *   `dim` 인자는 softmax를 적용할 차원을 지정합니다. (예: `dim=1`은 각 행에 대해 softmax 적용)
*   **`nn.LogSoftmax(dim=None)` / `F.log_softmax(input, dim=None)`**:
    *   Softmax 결과에 로그를 취한 값입니다. `NLLLoss` (Negative Log Likelihood Loss)와 함께 사용되어 수치적 안정성과 계산 효율성을 높입니다.

이 외에도 `ELU`, `SELU`, `SiLU (Swish)`, `GELU` 등 다양한 활성화 함수가 있습니다.

---

## 5. 손실 함수 (Loss Functions)

손실 함수(또는 비용 함수, 목적 함수)는 모델의 예측 값과 실제 정답 간의 차이를 측정하는 함수입니다. 모델은 이 손실 함수의 값을 최소화하는 방향으로 학습됩니다. `torch.nn` 모듈로 제공됩니다.

*   **`nn.MSELoss(reduction='mean')` (Mean Squared Error Loss)**:
    *   회귀(regression) 문제에 주로 사용됩니다. 예측값과 실제값 차이의 제곱 평균을 계산합니다.
    *   `reduction` 인자: `'mean'` (기본값, 평균), `'sum'` (합), `'none'` (각 요소별 손실 반환).
    ```python
    loss_fn = nn.MSELoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    loss = loss_fn(input, target)
    loss.backward()
    ```
*   **`nn.CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-100)`**:
    *   다중 클래스 분류(multi-class classification) 문제에 주로 사용됩니다.
    *   내부적으로 `LogSoftmax`와 `NLLLoss`를 결합한 형태이므로, 모델의 출력은 raw logits (Softmax를 거치지 않은 값) 형태여야 합니다.
    *   입력(예측): `(N, C)` 모양의 텐서 (N: 배치 크기, C: 클래스 개수). 값은 raw logits.
    *   타겟(정답): `(N)` 모양의 텐서. 각 값은 `0`부터 `C-1` 사이의 클래스 인덱스 (정수형).
    *   `weight`: 각 클래스에 대한 가중치를 지정하는 1D 텐서. 클래스 불균형 문제에 유용.
    *   `ignore_index`: 특정 타겟 값을 손실 계산에서 제외합니다.
    ```python
    loss_fn = nn.CrossEntropyLoss()
    input_logits = torch.randn(3, 5, requires_grad=True) # 3개의 샘플, 5개의 클래스
    target_labels = torch.empty(3, dtype=torch.long).random_(5) # 각 샘플의 정답 클래스 인덱스
    loss = loss_fn(input_logits, target_labels)
    loss.backward()
    ```
*   **`nn.BCELoss(reduction='mean')` (Binary Cross Entropy Loss)**:
    *   이진 분류(binary classification) 또는 다중 레이블 분류(multi-label classification) 문제에 사용됩니다.
    *   입력(예측): `0`과 `1` 사이의 확률 값 (일반적으로 Sigmoid 함수의 출력).
    *   타겟(정답): `0` 또는 `1`의 값.
*   **`nn.BCEWithLogitsLoss(reduction='mean', pos_weight=None)`**:
    *   `BCELoss`와 `Sigmoid` 계층을 결합한 형태로, 수치적 안정성이 더 좋습니다. 모델의 출력은 raw logits여야 합니다.
    *   `pos_weight`: 양성 샘플에 대한 가중치. 이진 분류에서 클래스 불균형이 있을 때 유용.
*   **`nn.NLLLoss(reduction='mean')` (Negative Log Likelihood Loss)**:
    *   `LogSoftmax`의 출력과 함께 사용되어 다중 클래스 분류 손실을 계산합니다. `CrossEntropyLoss`가 더 일반적으로 사용됩니다.
*   **기타 손실 함수**: `L1Loss` (Mean Absolute Error), `SmoothL1Loss`, `KLDivLoss` (Kullback-Leibler divergence), `MarginRankingLoss`, `HingeEmbeddingLoss`, `TripletMarginLoss` 등 다양한 손실 함수가 특정 작업에 맞게 제공됩니다.

---

## 6. 옵티마이저 (`torch.optim`)

옵티마이저는 계산된 손실 함수(loss)와 기울기(gradient)를 사용하여 모델의 학습 가능한 파라미터(가중치와 편향)를 업데이트하는 알고리즘입니다. `torch.optim` 패키지에 다양한 최적화 알고리즘이 구현되어 있습니다.

*   **옵티마이저 생성**: 옵티마이저를 생성할 때는 업데이트할 모델의 파라미터(`model.parameters()`)와 학습률(`lr`)을 전달합니다.
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    ```
*   **주요 옵티마이저**:
    *   **`optim.SGD` (Stochastic Gradient Descent)**: 가장 기본적인 경사 하강법 알고리즘. `momentum`, `weight_decay` (L2 정규화), `nesterov` 등의 옵션을 통해 개선 가능.
    *   **`optim.Adam` (Adaptive Moment Estimation)**: 각 파라미터마다 학습률을 적응적으로 조절하는 알고리즘. 일반적으로 좋은 성능을 보이며 많이 사용됩니다. `betas` (지수 이동 평균의 감쇠율), `eps` (수치 안정성을 위한 작은 값), `weight_decay` 등의 파라미터를 가집니다.
    *   **`optim.AdamW`**: Adam에 decoupled weight decay를 적용하여 정규화 성능을 개선한 버전.
    *   **`optim.RMSprop`**: 학습률을 적응적으로 조절하는 또 다른 알고리즘.
    *   **`optim.Adagrad`**: 학습률을 각 파라미터가 얼마나 자주 업데이트되었는지에 따라 조절. 자주 업데이트되지 않은 파라미터는 더 큰 학습률을 가짐.
    *   **`optim.Adadelta`**: Adagrad의 학습률이 계속 감소하는 문제를 개선.
*   **학습 과정에서의 옵티마이저 사용법**:
    1.  **기울기 초기화**: 매 학습 스텝 시작 시, 이전 스텝에서 계산된 기울기를 초기화합니다.
        ```python
        optimizer.zero_grad()
        # 또는 model.zero_grad() (모델 파라미터의 grad 속성을 직접 0으로 설정)
        # 또는 param.grad = None (더 효율적일 수 있음)
        ```
    2.  **순전파 및 손실 계산**: 모델에 입력을 전달하여 예측을 얻고, 이 예측과 실제 정답을 사용하여 손실을 계산합니다.
        ```python
        outputs = model(inputs)
        loss = criterion(outputs, labels) # criterion은 손실 함수 객체
        ```
    3.  **역전파**: 손실에 대해 `.backward()`를 호출하여 기울기를 계산합니다.
        ```python
        loss.backward()
        ```
    4.  **파라미터 업데이트**: 옵티마이저의 `step()` 메소드를 호출하여 모델의 파라미터를 업데이트합니다.
        ```python
        optimizer.step()
        ```

### 학습률 스케줄링 (`torch.optim.lr_scheduler`)

학습 과정 동안 학습률을 동적으로 조절하는 것은 모델 성능 향상에 도움이 될 수 있습니다. `torch.optim.lr_scheduler`는 다양한 학습률 스케줄링 방법을 제공합니다.
예시: `StepLR` (일정 epoch마다 학습률 감소), `ReduceLROnPlateau` (검증 손실이 더 이상 개선되지 않을 때 학습률 감소).

```python
from torch.optim.lr_scheduler import StepLR
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1) # 30 에폭마다 학습률에 0.1 곱함

for epoch in range(100):
    # train(...)
    # validate(...)
    scheduler.step() # 에폭 단위로 스케줄러 업데이트
```

---

## 7. 장치 할당 (`.to(device)`)

PyTorch는 텐서와 모델을 CPU 또는 GPU(CUDA, MPS 등)와 같은 다른 장치로 쉽게 옮길 수 있도록 지원합니다.

*   **장치 지정**:
    ```python
    device = torch.device('cuda' if torch.cuda.is_available() else \
                          'mps' if torch.backends.mps.is_available() else \
                          'cpu')
    print(f"Using {device} device")
    ```
    *   `torch.cuda.is_available()`: 사용 가능한 CUDA GPU가 있는지 확인합니다.
    *   `torch.backends.mps.is_available()`: Apple Silicon GPU (MPS)가 사용 가능한지 확인합니다.
*   **텐서 및 모델 이동**:
    *   `tensor.to(device)` 또는 `tensor.cuda(device_id)`: 텐서를 지정된 장치로 이동시킵니다. (새로운 텐서 반환)
    *   `model.to(device)` 또는 `model.cuda(device_id)`: 모델의 모든 파라미터와 버퍼를 지정된 장치로 이동시킵니다. (모델 자체를 변경, in-place)
    ```python
    tensor_on_device = tensor_on_cpu.to(device)
    model.to(device) # 모델의 모든 파라미터를 device로 이동
    ```
*   **주의사항**:
    *   **연산은 동일 장치에서**: 서로 다른 장치에 있는 텐서들 간의 연산은 직접적으로 수행할 수 없습니다. 연산을 수행하기 전에 모든 관련 텐서를 동일한 장치로 옮겨야 합니다.
    *   **데이터 로더**: `DataLoader`에서 데이터를 가져올 때, 각 배치 데이터를 적절한 장치로 옮겨야 합니다.

```python
# 일반적인 훈련 루프에서의 장치 할당
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(inputs) # 모델과 입력이 같은 장치에 있어야 함
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

---

## 8. 가중치 초기화 (Weight Initialization)

신경망의 가중치를 어떻게 초기화하느냐는 모델의 학습 속도와 최종 성능에 큰 영향을 미칠 수 있습니다. `torch.nn.init` 모듈은 다양한 가중치 초기화 방법을 제공합니다.

*   **기본 초기화**: PyTorch의 각 레이어(`nn.Linear`, `nn.Conv2d` 등)는 내부적으로 합리적인 기본 초기화 방법을 사용합니다. (예: `nn.Linear`는 Kaiming Uniform 초기화를 사용)
*   **수동 초기화**: 특정 초기화 전략을 사용하고 싶을 때 `torch.nn.init`의 함수들을 사용할 수 있습니다.
    *   `init.uniform_(tensor, a=0.0, b=1.0)`: 균등 분포.
    *   `init.normal_(tensor, mean=0.0, std=1.0)`: 정규 분포.
    *   `init.constant_(tensor, val)`: 상수 값으로 초기화.
    *   `init.zeros_(tensor)`: 0으로 초기화.
    *   `init.ones_(tensor)`: 1로 초기화.
    *   `init.xavier_uniform_(tensor, gain=1.0)` (Glorot Uniform): 입력과 출력의 분산을 동일하게 유지하려는 초기화. Sigmoid, Tanh 활성화 함수와 잘 어울림.
    *   `init.xavier_normal_(tensor, gain=1.0)` (Glorot Normal).
    *   `init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')` (He Uniform): ReLU 계열 활성화 함수와 잘 어울림. `mode` ('fan_in' 또는 'fan_out')는 분산 계산 시 고려할 연결 수.
    *   `init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')` (He Normal).
    *   `init.orthogonal_(tensor, gain=1.0)`: 직교 행렬로 초기화. RNN에서 유용할 수 있음.

*   **모델 전체에 초기화 적용**: `model.modules()`를 순회하며 각 레이어 타입에 맞게 초기화 함수를 적용할 수 있습니다. `model.apply(fn)`를 사용하면 더 간결하게 적용 가능합니다.
    ```python
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model = SimpleModel()
    model.apply(weights_init) # 모델의 모든 서브 모듈에 weights_init 함수 적용
    ```
    *   `@torch.no_grad()` 데코레이터나 컨텍스트 매니저를 초기화 함수에 사용하면, 초기화 과정 자체가 계산 그래프에 포함되지 않도록 할 수 있습니다. (보통 권장됨)

올바른 가중치 초기화는 기울기 소실 또는 폭발(exploding) 문제를 방지하고, 모델이 더 빠르고 안정적으로 수렴하도록 돕습니다. 