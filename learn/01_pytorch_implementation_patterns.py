# learn/01_pytorch_implementation_patterns.py

# PyTorch 핵심 구현 패턴 (Key Implementation Patterns)

# 이 파일은 PyTorch에서 자주 사용되는 핵심 구현 패턴들을 다룹니다.
# 예: nn.Module 상속, Dataset 커스텀, DataLoader, 모델 저장/로드 등

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

print("=== PyTorch 핵심 구현 패턴 (Key Implementation Patterns) ===\n")

# 1. nn.Module 상속 패턴
print("--- 1. nn.Module 상속 패턴 ---")

# 패턴 1: 기본적인 nn.Module 상속
class BasicLinearModel(nn.Module):
    """가장 기본적인 nn.Module 상속 패턴"""
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicLinearModel, self).__init__()  # 또는 super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 패턴 2: 조건부 레이어를 포함한 모델
class ConditionalModel(nn.Module):
    """조건부 레이어 및 드롭아웃을 포함한 패턴"""
    def __init__(self, input_size, hidden_size, output_size, use_dropout=True, dropout_p=0.5):
        super(ConditionalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        if self.use_dropout:
            x = self.dropout(x)
            
        x = self.fc2(x)
        x = self.relu(x)
        
        if self.use_dropout:
            x = self.dropout(x)
            
        x = self.fc3(x)
        return x

# 패턴 3: ModuleList를 사용한 동적 레이어 구성
class DynamicModel(nn.Module):
    """ModuleList를 사용한 동적 레이어 구성 패턴"""
    def __init__(self, layer_sizes):
        super(DynamicModel, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
        self.relu = nn.ReLU()
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 마지막 레이어가 아니면 활성화 함수 적용
            if i < len(self.layers) - 1:
                x = self.relu(x)
        return x

# 모델 인스턴스 생성 및 테스트
basic_model = BasicLinearModel(10, 20, 5)
conditional_model = ConditionalModel(10, 20, 5, use_dropout=True)
dynamic_model = DynamicModel([10, 20, 15, 5])

print(f"BasicLinearModel: {basic_model}")
print(f"ConditionalModel: {conditional_model}")
print(f"DynamicModel: {dynamic_model}")

# 입력 데이터로 테스트
test_input = torch.randn(3, 10)  # batch_size=3, input_size=10
print(f"\nTest input shape: {test_input.shape}")
print(f"BasicLinearModel output shape: {basic_model(test_input).shape}")
print(f"ConditionalModel output shape: {conditional_model(test_input).shape}")
print(f"DynamicModel output shape: {dynamic_model(test_input).shape}")


# 2. Dataset 커스텀 패턴
print("\n--- 2. Dataset 커스텀 패턴 ---")

class SimpleDataset(Dataset):
    """가장 기본적인 Dataset 상속 패턴"""
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, target

class NumpyDataset(Dataset):
    """NumPy 배열을 위한 Dataset 패턴"""
    def __init__(self, data_array, target_array, dtype=torch.float32):
        self.data = torch.from_numpy(data_array).type(dtype)
        self.targets = torch.from_numpy(target_array).type(dtype)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class SyntheticDataset(Dataset):
    """합성 데이터를 생성하는 Dataset 패턴"""
    def __init__(self, num_samples, input_dim, noise_std=0.1):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.noise_std = noise_std
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 간단한 선형 관계 y = 2*x + 1 + noise
        x = torch.randn(self.input_dim)
        y = 2 * x.sum() + 1 + torch.randn(1) * self.noise_std
        return x, y

# Dataset 인스턴스 생성 및 테스트
# 더미 데이터 생성
dummy_data = np.random.randn(100, 10)
dummy_targets = np.random.randn(100, 1)

simple_dataset = SimpleDataset(dummy_data, dummy_targets)
numpy_dataset = NumpyDataset(dummy_data, dummy_targets)
synthetic_dataset = SyntheticDataset(num_samples=100, input_dim=10)

print(f"SimpleDataset length: {len(simple_dataset)}")
print(f"NumpyDataset length: {len(numpy_dataset)}")
print(f"SyntheticDataset length: {len(synthetic_dataset)}")

# 샘플 확인
sample_data, sample_target = simple_dataset[0]
print(f"SimpleDataset sample - data shape: {sample_data.shape if hasattr(sample_data, 'shape') else type(sample_data)}, target shape: {sample_target.shape if hasattr(sample_target, 'shape') else type(sample_target)}")

sample_data, sample_target = numpy_dataset[0]
print(f"NumpyDataset sample - data shape: {sample_data.shape}, target shape: {sample_target.shape}")

sample_data, sample_target = synthetic_dataset[0]
print(f"SyntheticDataset sample - data shape: {sample_data.shape}, target shape: {sample_target.shape}")


# 3. DataLoader 사용법 패턴
print("\n--- 3. DataLoader 사용법 패턴 ---")

# 기본 DataLoader 사용
dataloader = DataLoader(numpy_dataset, batch_size=16, shuffle=True, num_workers=0)

print(f"DataLoader batch size: {dataloader.batch_size}")
print(f"DataLoader dataset size: {len(dataloader.dataset)}")
print(f"Number of batches: {len(dataloader)}")

# DataLoader 반복 패턴
print("\nDataLoader 반복 패턴:")
for batch_idx, (data, target) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
    if batch_idx >= 2:  # 처음 3개 배치만 출력
        break

# 커스텀 collate_fn 패턴
def custom_collate_fn(batch):
    """커스텀 배치 처리 함수"""
    data, targets = zip(*batch)
    data = torch.stack(data)
    targets = torch.stack(targets)
    
    # 예시: 데이터 정규화
    data = (data - data.mean()) / (data.std() + 1e-8)
    
    return data, targets

custom_dataloader = DataLoader(
    numpy_dataset, 
    batch_size=8, 
    shuffle=True, 
    collate_fn=custom_collate_fn,
    num_workers=0
)

print(f"\nCustom DataLoader with collate_fn:")
for batch_idx, (data, target) in enumerate(custom_dataloader):
    print(f"Batch {batch_idx}: data mean {data.mean():.4f}, data std {data.std():.4f}")
    if batch_idx >= 1:  # 처음 2개 배치만 출력
        break


# 4. train() / eval() 모드와 no_grad() 패턴
print("\n--- 4. train() / eval() 모드와 no_grad() 패턴 ---")

# train() / eval() 모드 차이 확인
model_with_dropout = ConditionalModel(10, 20, 5, use_dropout=True, dropout_p=0.5)

print("=== train() 모드 ===")
model_with_dropout.train()  # 훈련 모드로 설정
print(f"Model training mode: {model_with_dropout.training}")

test_input = torch.randn(5, 10)
output1 = model_with_dropout(test_input)
output2 = model_with_dropout(test_input)  # 같은 입력이지만 드롭아웃으로 인해 다른 출력

print(f"Train mode - Output1 mean: {output1.mean():.4f}")
print(f"Train mode - Output2 mean: {output2.mean():.4f}")
print(f"Outputs are different due to dropout: {not torch.allclose(output1, output2)}")

print("\n=== eval() 모드 ===")
model_with_dropout.eval()  # 평가 모드로 설정
print(f"Model training mode: {model_with_dropout.training}")

output3 = model_with_dropout(test_input)
output4 = model_with_dropout(test_input)  # 같은 입력, 드롭아웃 비활성화로 같은 출력

print(f"Eval mode - Output3 mean: {output3.mean():.4f}")
print(f"Eval mode - Output4 mean: {output4.mean():.4f}")
print(f"Outputs are the same (no dropout): {torch.allclose(output3, output4)}")

# no_grad() 패턴
print("\n=== torch.no_grad() 패턴 ===")
model_with_dropout.train()
input_with_grad = torch.randn(5, 10, requires_grad=True)

# 기울기 추적 O
output_with_grad = model_with_dropout(input_with_grad)
print(f"With grad tracking - output requires_grad: {output_with_grad.requires_grad}")
print(f"Input grad_fn: {input_with_grad.grad_fn}")
print(f"Output grad_fn: {output_with_grad.grad_fn}")

# 기울기 추적 X
with torch.no_grad():
    output_no_grad = model_with_dropout(input_with_grad)
    print(f"With no_grad - output requires_grad: {output_no_grad.requires_grad}")
    print(f"With no_grad - output grad_fn: {output_no_grad.grad_fn}")


# 5. 모델 저장/로드 패턴
print("\n--- 5. 모델 저장/로드 패턴 ---")

# 훈련된 모델 생성 (간단한 훈련 시뮬레이션)
model_to_save = BasicLinearModel(10, 20, 5)
optimizer = optim.Adam(model_to_save.parameters(), lr=0.001)

# 간단한 훈련 루프 시뮬레이션
model_to_save.train()
for epoch in range(3):
    optimizer.zero_grad()
    dummy_input = torch.randn(32, 10)
    dummy_target = torch.randn(32, 5)
    output = model_to_save(dummy_input)
    loss = F.mse_loss(output, dummy_target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 패턴 1: 전체 모델 저장/로드
print("\n=== 패턴 1: 전체 모델 저장/로드 ===")
model_path = "temp_full_model.pth"
torch.save(model_to_save, model_path)
print(f"Full model saved to {model_path}")

loaded_model = torch.load(model_path)
print(f"Model loaded successfully")
print(f"Original model type: {type(model_to_save)}")
print(f"Loaded model type: {type(loaded_model)}")

# 패턴 2: state_dict 저장/로드 (권장 방법)
print("\n=== 패턴 2: state_dict 저장/로드 (권장) ===")
state_dict_path = "temp_state_dict.pth"

# 모델 상태 저장
torch.save(model_to_save.state_dict(), state_dict_path)
print(f"Model state_dict saved to {state_dict_path}")

# 새 모델 인스턴스 생성 후 state_dict 로드
new_model = BasicLinearModel(10, 20, 5)
new_model.load_state_dict(torch.load(state_dict_path))
print(f"State dict loaded into new model")

# 모델 파라미터 비교
original_params = list(model_to_save.parameters())
loaded_params = list(new_model.parameters())

params_equal = all(torch.allclose(p1, p2) for p1, p2 in zip(original_params, loaded_params))
print(f"Parameters are identical: {params_equal}")

# 패턴 3: 체크포인트 저장/로드 (모델 + 옵티마이저 + 기타 정보)
print("\n=== 패턴 3: 체크포인트 저장/로드 ===")
checkpoint_path = "temp_checkpoint.pth"

# 체크포인트 저장
checkpoint = {
    'epoch': 3,
    'model_state_dict': model_to_save.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    'model_config': {
        'input_size': 10,
        'hidden_size': 20,
        'output_size': 5
    }
}

torch.save(checkpoint, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")

# 체크포인트 로드
checkpoint = torch.load(checkpoint_path)
restored_model = BasicLinearModel(
    checkpoint['model_config']['input_size'],
    checkpoint['model_config']['hidden_size'],
    checkpoint['model_config']['output_size']
)
restored_optimizer = optim.Adam(restored_model.parameters(), lr=0.001)

restored_model.load_state_dict(checkpoint['model_state_dict'])
restored_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
last_loss = checkpoint['loss']

print(f"Checkpoint loaded - Epoch: {start_epoch}, Last loss: {last_loss:.4f}")

# 패턴 4: 조건부 로드 (GPU/CPU 호환성)
print("\n=== 패턴 4: 조건부 로드 (GPU/CPU 호환성) ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

# 장치에 맞는 로드
if device.type == 'cuda':
    # GPU에서 로드
    loaded_checkpoint = torch.load(checkpoint_path)
else:
    # CPU에서 로드 (GPU에서 저장된 모델도 CPU로 로드 가능)
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)

print(f"Checkpoint loaded on {device}")

# 임시 파일 정리
import os
temp_files = [model_path, state_dict_path, checkpoint_path]
for file_path in temp_files:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Cleaned up: {file_path}")


# 6. 일반적인 훈련 루프 패턴
print("\n--- 6. 일반적인 훈련 루프 패턴 ---")

def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 훈련을 수행하는 함수"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # 기울기 초기화
        optimizer.zero_grad()
        
        # 순전파
        output = model(data)
        
        # 손실 계산
        loss = criterion(output, target)
        
        # 역전파
        loss.backward()
        
        # 파라미터 업데이트
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_model(model, dataloader, criterion, device):
    """모델 평가를 수행하는 함수"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

# 훈련 루프 실행 예시
print("=== 훈련 루프 패턴 실행 ===")
device = torch.device('cpu')  # 예시를 위해 CPU 사용
model = BasicLinearModel(10, 20, 1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 데이터셋 및 데이터로더 생성
train_dataset = SyntheticDataset(num_samples=1000, input_dim=10)
val_dataset = SyntheticDataset(num_samples=200, input_dim=10)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 훈련 실행
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate_model(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print("\nPyTorch 핵심 구현 패턴 스크립트 실행 완료!") 