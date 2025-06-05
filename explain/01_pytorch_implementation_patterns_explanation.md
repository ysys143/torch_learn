# PyTorch 핵심 구현 패턴 (Key Implementation Patterns) 설명

## 개요

PyTorch는 딥러닝 개발을 위한 강력한 프레임워크이지만, 효과적으로 사용하기 위해서는 몇 가지 핵심 구현 패턴을 숙지해야 합니다. 이 문서는 PyTorch 개발에서 반드시 알아야 할 주요 패턴들을 설명합니다.

## 1. nn.Module 상속 패턴

### 1.1 기본 원리

`nn.Module`은 PyTorch에서 모든 신경망 구성 요소의 기본 클래스입니다. 이를 상속받아 커스텀 모델을 구현할 때 다음 원칙을 따라야 합니다:

- **`__init__()` 메서드**: 모든 레이어와 파라미터를 정의
- **`forward()` 메서드**: 순전파 로직을 구현
- **`super().__init__()` 호출**: 부모 클래스 초기화 필수

### 1.2 구현 패턴별 특징

#### 기본 패턴
```python
class BasicLinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

**특징:**
- 명확한 레이어 분리
- 순차적 데이터 흐름
- 간단하고 이해하기 쉬운 구조

#### 조건부 패턴
```python
class ConditionalModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_dropout=True):
        super(ConditionalModel, self).__init__()
        # 레이어 정의
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)
```

**장점:**
- 런타임 조건에 따른 모델 동작 변경
- 하이퍼파라미터를 통한 아키텍처 제어
- 실험 및 비교 연구에 유용

#### 동적 패턴 (ModuleList)
```python
class DynamicModel(nn.Module):
    def __init__(self, layer_sizes):
        super(DynamicModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
```

**활용 사례:**
- 가변 길이 아키텍처
- AutoML에서 자동 아키텍처 탐색
- 실험적 모델 구조 테스트

### 1.3 주의사항

1. **파라미터 등록**: `nn.Module`을 상속받지 않은 레이어는 자동으로 파라미터가 등록되지 않음
2. **디바이스 이동**: 모델을 GPU로 이동할 때 모든 서브모듈이 함께 이동됨
3. **모드 전환**: `train()`과 `eval()` 모드가 모든 서브모듈에 적용됨

## 2. Dataset 커스텀 패턴

### 2.1 기본 구조

PyTorch의 `Dataset` 클래스는 다음 메서드를 반드시 구현해야 합니다:

- `__len__()`: 데이터셋 크기 반환
- `__getitem__(idx)`: 인덱스에 해당하는 샘플 반환

### 2.2 구현 패턴별 특징

#### 메모리 로드 패턴
```python
class NumpyDataset(Dataset):
    def __init__(self, data_array, target_array):
        self.data = torch.from_numpy(data_array)
        self.targets = torch.from_numpy(target_array)
```

**특징:**
- 작은 데이터셋에 적합
- 빠른 데이터 접근
- 메모리 사용량 높음

#### 지연 로드 패턴
```python
class LazyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        
    def __getitem__(self, idx):
        # 실제 사용 시점에 파일 로드
        return load_file(self.file_paths[idx])
```

**특징:**
- 대용량 데이터셋에 적합
- 메모리 효율적
- I/O 오버헤드 존재

#### 합성 데이터 패턴
```python
class SyntheticDataset(Dataset):
    def __getitem__(self, idx):
        # 동적으로 데이터 생성
        x = torch.randn(input_dim)
        y = some_function(x)
        return x, y
```

**활용 사례:**
- 무한 데이터 생성
- 시뮬레이션 데이터
- 데이터 증강

### 2.3 변환(Transform) 패턴

```python
class TransformDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.targets[idx]
```

**장점:**
- 데이터 전처리 표준화
- 재사용 가능한 변환 파이프라인
- 실시간 데이터 증강

## 3. DataLoader 사용법 패턴

### 3.1 기본 사용법

```python
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True  # GPU 사용 시 권장
)
```

### 3.2 주요 매개변수

- **batch_size**: 배치 크기 (메모리와 성능의 균형점)
- **shuffle**: 데이터 섞기 (훈련 시 True, 평가 시 False)
- **num_workers**: 병렬 데이터 로딩 프로세스 수
- **pin_memory**: GPU 전송 최적화 (GPU 사용 시 True)
- **drop_last**: 마지막 불완전 배치 제거 여부

### 3.3 커스텀 collate_fn 패턴

```python
def custom_collate_fn(batch):
    data, targets = zip(*batch)
    data = torch.stack(data)
    targets = torch.stack(targets)
    
    # 커스텀 전처리
    data = normalize(data)
    
    return data, targets
```

**사용 사례:**
- 가변 길이 시퀀스 처리
- 배치 단위 정규화
- 복잡한 데이터 구조 처리

### 3.4 성능 최적화 팁

1. **num_workers 조정**: CPU 코어 수에 맞게 설정
2. **pin_memory 활용**: GPU 사용 시 메모리 전송 속도 향상
3. **prefetch_factor**: 미리 로드할 배치 수 조정
4. **persistent_workers**: 워커 프로세스 재사용

## 4. train() / eval() 모드와 no_grad() 패턴

### 4.1 훈련 모드 vs 평가 모드

#### train() 모드
- Dropout, BatchNorm 등이 활성화
- 파라미터 업데이트 가능
- 기울기 계산 수행

```python
model.train()
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

#### eval() 모드
- Dropout 비활성화, BatchNorm은 통계값 고정
- 추론 모드로 동작
- 일관된 예측 결과

```python
model.eval()
with torch.no_grad():
    for data, target in val_loader:
        output = model(data)
        # 손실 계산 및 메트릭 수집
```

### 4.2 torch.no_grad() 패턴

#### 기울기 계산 비활성화
```python
with torch.no_grad():
    output = model(input)
```

**효과:**
- 메모리 사용량 감소
- 계산 속도 향상
- 추론 시 필수 사용

#### 컨텍스트 매니저 vs 데코레이터
```python
# 컨텍스트 매니저
with torch.no_grad():
    result = model(input)

# 데코레이터
@torch.no_grad()
def inference_function(model, input):
    return model(input)
```

### 4.3 주의사항

1. **모드 일관성**: 훈련과 평가 시 올바른 모드 설정 필수
2. **중첩된 모듈**: 모든 서브모듈에 모드가 적용됨
3. **배치 정규화**: eval() 모드에서 이동 평균 통계 사용

## 5. 모델 저장/로드 패턴

### 5.1 저장 방법 비교

#### 전체 모델 저장
```python
torch.save(model, 'model.pth')
loaded_model = torch.load('model.pth')
```

**장점:** 간단함
**단점:** 
- 모델 클래스 정의 필요
- 버전 호환성 문제
- 파일 크기 큼

#### state_dict 저장 (권장)
```python
torch.save(model.state_dict(), 'model_state.pth')
model = MyModel()
model.load_state_dict(torch.load('model_state.pth'))
```

**장점:**
- 호환성 좋음
- 파일 크기 작음
- 모델 구조 변경에 유연

#### 체크포인트 저장
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': config
}
torch.save(checkpoint, 'checkpoint.pth')
```

**활용:**
- 훈련 중단/재시작
- 최적 모델 추적
- 실험 재현성

### 5.2 장치 호환성

#### CPU/GPU 간 로드
```python
# GPU에서 저장, CPU에서 로드
model = torch.load('model.pth', map_location='cpu')

# 현재 장치로 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pth', map_location=device)
```

### 5.3 버전 관리 팁

1. **설정 저장**: 모델 하이퍼파라미터 함께 저장
2. **버전 태그**: PyTorch 버전 정보 포함
3. **검증**: 로드 후 빠른 추론 테스트
4. **백업**: 여러 체크포인트 유지

## 6. 일반적인 훈련 루프 패턴

### 6.1 표준 훈련 루프

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()    # 기울기 초기화
        output = model(data)     # 순전파
        loss = criterion(output, target)  # 손실 계산
        loss.backward()          # 역전파
        optimizer.step()         # 파라미터 업데이트
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 6.2 평가 루프

```python
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 6.3 고급 훈련 패턴

#### 기울기 누적
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 기울기 클리핑
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

#### 혼합 정밀도 훈련
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 7. 모범 사례 및 팁

### 7.1 코드 구조화

1. **관심사 분리**: 모델, 데이터, 훈련 로직 분리
2. **설정 관리**: 하이퍼파라미터를 별도 파일로 관리
3. **로깅**: 훈련 과정 추적 및 시각화
4. **재현성**: 랜덤 시드 고정

### 7.2 성능 최적화

1. **배치 크기**: GPU 메모리에 맞는 최대 크기 사용
2. **데이터 로딩**: 병렬 처리 및 프리페치 활용
3. **모델 병렬화**: 다중 GPU 활용
4. **메모리 관리**: 불필요한 기울기 계산 방지

### 7.3 디버깅 팁

1. **작은 데이터셋**: 오버피팅 테스트
2. **기울기 확인**: 기울기 소실/폭발 모니터링
3. **중간 출력**: 모델 각 단계 출력 확인
4. **프로파일링**: 병목 지점 식별

## 결론

PyTorch의 핵심 구현 패턴을 숙지하면 효율적이고 유지보수가 쉬운 딥러닝 코드를 작성할 수 있습니다. 각 패턴의 장단점을 이해하고, 상황에 맞는 적절한 패턴을 선택하는 것이 중요합니다. 또한 지속적인 실습을 통해 이러한 패턴들을 자연스럽게 활용할 수 있도록 연습하는 것을 권장합니다. 