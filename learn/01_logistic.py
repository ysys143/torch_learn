"""
PyTorch 로지스틱 회귀 튜토리얼
=============================

이 튜토리얼에서는 PyTorch를 사용하여 간단한 로지스틱 회귀 모델을 구현합니다.
타이타닉 데이터셋을 사용하여 승객의 생존 여부를 예측해보겠습니다.

주요 단계:
1. 데이터 로드 및 전처리
2. 로지스틱 회귀 모델 정의
3. 모델 훈련 (피팅)
4. 모델 평가 및 추론
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 환경변수 로드
import os
from dotenv import load_dotenv
load_dotenv()

print("=== PyTorch 로지스틱 회귀 튜토리얼 ===\n")

# 1. 데이터 로드 및 전처리
print("1. 데이터 로드 중...")

# 타이타닉 데이터셋 URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

try:
    # 데이터 로드
    df = pd.read_csv(url)
    print(f"데이터 로드 완료! 데이터 크기: {df.shape}")
    print(f"컬럼: {list(df.columns)}\n")
    
    # 기본 정보 출력
    print("데이터 미리보기:")
    print(df.head())
    print("\n데이터 정보:")
    print(df.info())
    print(f"\n생존률: {df['Survived'].mean():.2%}")
    
except Exception as e:
    print(f"데이터 로드 실패: {e}")
    # 백업: 간단한 합성 데이터 생성
    print("합성 데이터를 생성합니다...")
    np.random.seed(42)
    n_samples = 800
    df = pd.DataFrame({
        'Age': np.random.normal(30, 12, n_samples),
        'Fare': np.random.exponential(20, n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'Sex_male': np.random.choice([0, 1], n_samples),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })

print("\n2. 데이터 전처리...")

# 필요한 피처 선택 및 전처리
if 'Sex' in df.columns:
    # 성별을 숫자로 변환 (male=1, female=0)
    df['Sex_male'] = (df['Sex'] == 'male').astype(int)

# 사용할 피처 선택
features = ['Pclass', 'Sex_male', 'Age', 'Fare']
available_features = [f for f in features if f in df.columns]

print(f"사용 가능한 피처: {available_features}")

# 결측값 처리
for feature in available_features:
    if df[feature].isnull().sum() > 0:
        df[feature].fillna(df[feature].median(), inplace=True)

# 특성과 타겟 분리
X = df[available_features].values
y = df['Survived'].values

print(f"입력 데이터 형태: {X.shape}")
print(f"타겟 데이터 형태: {y.shape}")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 특성 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PyTorch 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

print(f"훈련 데이터: {X_train_tensor.shape}")
print(f"테스트 데이터: {X_test_tensor.shape}")

# 2. 로지스틱 회귀 모델 정의
print("\n3. 로지스틱 회귀 모델 정의...")

class LogisticRegression(nn.Module):
    """
    간단한 로지스틱 회귀 모델
    
    로지스틱 회귀는 선형 결합 후 시그모이드 함수를 적용하여
    이진 분류를 수행하는 모델입니다.
    """
    
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        # 선형 레이어: 입력 특성들을 하나의 값으로 변환
        self.linear = nn.Linear(input_size, 1)
        # 시그모이드 함수: 0과 1 사이의 확률값으로 변환
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
        # 선형 변환 후 시그모이드 적용
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

# 모델 인스턴스 생성
input_size = X_train_tensor.shape[1]
model = LogisticRegression(input_size)

print(f"모델 구조:")
print(model)
print(f"\n모델 파라미터:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 3. 모델 훈련 설정
print("\n4. 모델 훈련 설정...")

# 손실 함수: 이진 교차 엔트로피 (Binary Cross Entropy)
criterion = nn.BCELoss()

# 옵티마이저: Adam
# `weight_decay` (L2 정규화): 과적합(overfitting)을 방지하는 정규화 기법 중 하나입니다.
# L2 정규화는 모델의 가중치(weights)가 너무 커지는 것을 제어하여 모델의 복잡도를 줄입니다.
# 손실 함수에 가중치 제곱합의 일정 비율을 더함으로써 큰 가중치에 패널티를 부여합니다.
# `weight_decay=0.001`은 이 패널티의 강도를 조절하는 하이퍼파라미터입니다.
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

# 훈련 설정
num_epochs = 1000
losses = []

print(f"훈련 설정:")
print(f"- 에포크 수: {num_epochs}")
print(f"- 학습률: {optimizer.param_groups[0]['lr']}")
print(f"- 손실 함수: {criterion.__class__.__name__}")
print(f"- 옵티마이저: {optimizer.__class__.__name__}")
print(f"- L2 정규화 (Weight Decay): {optimizer.param_groups[0].get('weight_decay', 'N/A')}")

# 4. 모델 훈련 (피팅)
print("\n5. 모델 훈련 시작...")

model.train()  # 훈련 모드 설정

for epoch in range(num_epochs):
    # 순전파 (Forward pass)
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)
    
    # 역전파 (Backward pass)
    optimizer.zero_grad()  # 기울기 초기화
    loss.backward()        # 기울기 계산
    optimizer.step()       # 파라미터 업데이트
    
    # 손실 기록
    losses.append(loss.item())
    
    # 진행 상황 출력
    if (epoch + 1) % 100 == 0:
        print(f'에포크 [{epoch+1}/{num_epochs}], 손실: {loss.item():.4f}')

print("훈련 완료!")

# 5. 모델 평가 및 추론
print("\n6. 모델 평가...")

model.eval()  # 평가 모드 설정

with torch.no_grad():  # 기울기 계산 비활성화
    # 훈련 데이터 예측
    train_outputs = model(X_train_tensor)
    train_predictions = (train_outputs.squeeze() > 0.5).float()
    train_accuracy = accuracy_score(y_train_tensor, train_predictions)
    
    # 테스트 데이터 예측
    test_outputs = model(X_test_tensor)
    test_predictions = (test_outputs.squeeze() > 0.5).float()
    test_accuracy = accuracy_score(y_test_tensor, test_predictions)

print(f"훈련 정확도: {train_accuracy:.4f}")
print(f"테스트 정확도: {test_accuracy:.4f}")
