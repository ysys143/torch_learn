# learn/buggy_pytorch_example.py
# 이 파일은 의도적으로 버그를 포함하고 있습니다 - bugbot 테스트용

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 버그 1: 문법 오류 - 잘못된 들여쓰기
class BuggyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BuggyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    self.relu = nn.ReLU()  # 버그: 잘못된 들여쓰기
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # 버그 2: 변수명 오타
        x = self.fc2(X)  # X는 정의되지 않음
        return x

# 버그 3: 논리적 오류 - 잘못된 차원 계산
def create_data():
    # 입력 데이터: (batch_size, features)
    data = torch.randn(100, 10)
    # 버그: 타겟 차원이 입력과 맞지 않음
    targets = torch.randn(50, 5)  # 100이어야 하는데 50
    return data, targets

# 버그 4: 타입 오류
def buggy_loss_calculation(predictions, targets):
    # 버그: 문자열과 텐서 연산
    loss = "loss_value" + predictions.mean()
    return loss

# 버그 5: 잠재적 런타임 오류 - 0으로 나누기
def buggy_normalization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    # 버그: std가 0일 수 있음
    normalized = (tensor - mean) / std
    return normalized

# 버그 6: 메모리 누수 가능성
def memory_leak_training():
    model = BuggyModel(10, 20, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1000):
        data, targets = create_data()
        
        # 버그: optimizer.zero_grad() 누락
        # optimizer.zero_grad()  # 주석 처리됨
        
        output = model(data)
        
        # 버그: 잘못된 손실 함수 사용
        loss = buggy_loss_calculation(output, targets)
        
        # 버그: 문자열에 backward() 호출
        loss.backward()
        optimizer.step()
        
        # 버그: 메모리에 모든 손실값 저장 (메모리 누수)
        all_losses.append(loss)  # all_losses가 정의되지 않음

# 버그 7: 잘못된 데이터 타입 변환
def buggy_data_processing():
    # NumPy 배열을 생성
    np_array = np.random.randn(100, 10)
    
    # 버그: 잘못된 torch 변환
    torch_tensor = torch.tensor(np_array, dtype=torch.int32)  # float 데이터를 int로
    
    # 버그: GPU 메모리 부족 가능성
    huge_tensor = torch.randn(10000, 10000, 10000)  # 너무 큰 텐서
    
    return torch_tensor, huge_tensor

# 버그 8: 잘못된 모델 저장/로드
def buggy_save_load():
    model = BuggyModel(10, 20, 5)
    
    # 버그: 잘못된 파일 경로
    torch.save(model, "/invalid/path/model.pth")
    
    # 버그: 존재하지 않는 파일 로드
    loaded_model = torch.load("nonexistent_model.pth")
    
    return loaded_model

# 버그 9: 무한 루프 가능성
def potentially_infinite_loop():
    x = 1.0
    while x > 0:
        x = x * 1.001  # 버그: x가 계속 증가하여 무한 루프
        print(f"x = {x}")

# 버그 10: 잘못된 차원 조작
def dimension_bug():
    tensor = torch.randn(5, 10, 15)
    
    # 버그: 존재하지 않는 차원으로 reshape
    reshaped = tensor.reshape(3, 4, 5, 6, 7)  # 원소 수가 맞지 않음
    
    # 버그: 잘못된 인덱싱
    sliced = tensor[:, :, 20]  # 인덱스 20은 존재하지 않음
    
    return reshaped, sliced

# 버그 11: 사용되지 않는 import와 변수
import os  # 사용되지 않음
import sys  # 사용되지 않음
unused_variable = "이 변수는 사용되지 않습니다"

# 버그 12: 잘못된 함수 호출
if __name__ == "__main__":
    print("버그가 많은 코드 실행 중...")
    
    # 여러 버그가 있는 함수들 호출
    model = BuggyModel(10, 20, 5)
    data, targets = create_data()
    
    # 버그: 차원이 맞지 않는 연산
    output = model(data[:30])  # 30개 샘플
    loss = nn.MSELoss()(output, targets)  # targets는 50개
    
    print("실행 완료 (하지만 버그로 인해 실제로는 실행되지 않을 것입니다!)") 