"""
PyTorch ResNet 튜토리얼
======================

이 튜토리얼에서는 PyTorch를 사용하여 ResNet(Residual Network)을 구현하고,
CIFAR-10 이미지 데이터셋을 이용한 분류 과정을 시연합니다.

ResNet의 핵심 개념:
1. Residual Block (잔차 블록)
2. Skip Connection (건너뛰기 연결)
3. Identity Mapping (항등 매핑)
4. Vanishing Gradient 문제 해결

주요 단계:
1. 데이터 로드 및 전처리
2. ResNet 모델 정의 (잔차 블록 포함)
3. 모델 훈련 (피팅)
4. 모델 평가

SOLID 원칙 적용:
- Single Responsibility Principle: 각 클래스는 하나의 명확한 역할
- Open/Closed Principle: 확장에는 열려있고 수정에는 닫혀있는 구조
- Liskov Substitution Principle: 기본 블록 인터페이스 준수
- Interface Segregation Principle: 필요한 인터페이스만 의존
- Dependency Inversion Principle: 추상화에 의존하는 구조
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Optional, List
import os

print("=== PyTorch ResNet 튜토리얼 ===\n")


class ResidualBlock(nn.Module, ABC):
    """
    잔차 블록의 추상 기본 클래스
    
    Single Responsibility Principle: 잔차 연결의 기본 구조만 정의
    Open/Closed Principle: 상속을 통한 확장 가능, 수정에는 닫힘
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파 정의 (하위 클래스에서 구현)"""
        pass


class BasicBlock(ResidualBlock):
    """
    기본 잔차 블록 구현
    
    구조: 3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> (+shortcut) -> ReLU
    
    Single Responsibility Principle: 기본 잔차 블록의 구현에만 집중
    Liskov Substitution Principle: ResidualBlock을 완전히 대체 가능
    """
    
    expansion: int = 1  # 출력 채널 확장 비율
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        기본 잔차 블록 초기화
        
        Args:
            in_planes: 입력 채널 수
            planes: 중간 채널 수
            stride: 스트라이드 (다운샘플링용)
        """
        super().__init__()
        
        # 주 경로 (main path)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 숏컷 경로 (shortcut path) - 차원 맞추기용
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파: 잔차 연결 수행
        F(x) + x의 형태로 identity mapping 구현
        """
        # 주 경로
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 잔차 연결: F(x) + x
        out += self.shortcut(x)
        out = torch.relu(out)
        
        return out


class Bottleneck(ResidualBlock):
    """
    병목 잔차 블록 구현 (더 깊은 네트워크용)
    
    구조: 1x1 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 1x1 conv -> BN -> (+shortcut) -> ReLU
    
    Single Responsibility Principle: 병목 블록의 구현에만 집중
    Liskov Substitution Principle: ResidualBlock을 완전히 대체 가능
    """
    
    expansion: int = 4  # 출력 채널 확장 비율
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        병목 잔차 블록 초기화
        
        Args:
            in_planes: 입력 채널 수
            planes: 중간 채널 수 (최종 출력은 planes * expansion)
            stride: 스트라이드 (다운샘플링용)
        """
        super().__init__()
        
        # 주 경로: 1x1 -> 3x3 -> 1x1 구조
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        # 숏컷 경로
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파: 병목 구조의 잔차 연결 수행"""
        # 주 경로
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # 잔차 연결
        out += self.shortcut(x)
        out = torch.relu(out)
        
        return out


class ResNetBuilder:
    """
    ResNet 모델 생성을 위한 빌더 클래스
    
    Single Responsibility Principle: ResNet 구성에만 집중
    Dependency Inversion Principle: 추상 블록 타입에 의존
    """
    
    @staticmethod
    def create_layer(block_type: type[ResidualBlock], in_planes: int, planes: int, 
                    num_blocks: int, stride: int) -> nn.Sequential:
        """
        ResNet 레이어 생성
        
        Args:
            block_type: 사용할 잔차 블록 타입
            in_planes: 입력 채널 수
            planes: 중간 채널 수
            num_blocks: 블록 개수
            stride: 첫 번째 블록의 스트라이드
        
        Returns:
            nn.Sequential: 구성된 레이어
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block_type(in_planes, planes, stride))
            in_planes = planes * block_type.expansion
        
        return nn.Sequential(*layers)


class ResNet(nn.Module):
    """
    ResNet 모델 구현
    
    Single Responsibility Principle: ResNet 네트워크 구조 정의에만 집중
    Open/Closed Principle: 다양한 블록 타입으로 확장 가능
    Interface Segregation Principle: 필요한 기능만 노출
    """
    
    def __init__(self, block_type: type[ResidualBlock], num_blocks: List[int], 
                 num_classes: int = 10) -> None:
        """
        ResNet 모델 초기화
        
        Args:
            block_type: 사용할 잔차 블록 타입 (BasicBlock 또는 Bottleneck)
            num_blocks: 각 레이어의 블록 개수 리스트
            num_classes: 분류할 클래스 수
        """
        super().__init__()
        
        self.in_planes = 64
        
        # 초기 컨볼루션 레이어
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 잔차 레이어들
        self.layer1 = self._make_layer(block_type, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block_type, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block_type, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block_type, 512, num_blocks[3], 2)
        
        # 분류기
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)
        
        # 파라미터 초기화
        self._initialize_weights()
    
    def _make_layer(self, block_type: type[ResidualBlock], planes: int, 
                   num_blocks: int, stride: int) -> nn.Sequential:
        """
        ResNet 레이어 생성 (내부 메서드)
        
        Args:
            block_type: 사용할 잔차 블록 타입
            planes: 중간 채널 수
            num_blocks: 블록 개수
            stride: 첫 번째 블록의 스트라이드
        
        Returns:
            nn.Sequential: 구성된 레이어
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block_type(self.in_planes, planes, stride))
            self.in_planes = planes * block_type.expansion
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """
        가중치 초기화
        
        Single Responsibility Principle: 가중치 초기화에만 집중
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # 초기 컨볼루션
        out = torch.relu(self.bn1(self.conv1(x)))
        
        # 잔차 레이어들
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 분류
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


class ResNetFactory:
    """
    다양한 ResNet 변형을 생성하는 팩토리 클래스
    
    Single Responsibility Principle: ResNet 변형 생성에만 집중
    """
    
    @staticmethod
    def resnet18(num_classes: int = 10) -> ResNet:
        """ResNet-18 모델 생성"""
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    
    @staticmethod
    def resnet34(num_classes: int = 10) -> ResNet:
        """ResNet-34 모델 생성"""
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    
    @staticmethod
    def resnet50(num_classes: int = 10) -> ResNet:
        """ResNet-50 모델 생성"""
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    
    @staticmethod
    def resnet101(num_classes: int = 10) -> ResNet:
        """ResNet-101 모델 생성"""
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


# 1. 데이터 로드 및 전처리
print("1. CIFAR-10 데이터셋 로드 중...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"훈련 데이터셋 크기: {len(train_dataset)}")
print(f"테스트 데이터셋 크기: {len(test_dataset)}\n")

# 2. ResNet 모델 정의
print("2. ResNet-18 모델 정의...")

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
model = ResNetFactory.resnet18(num_classes=10).to(device)

print(f"모델 구조:")
print(f"- ResNet-18 (BasicBlock 사용)")
print(f"- 사용 디바이스: {device}")
print(f"- 총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
print(f"- 훈련 가능한 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

# 3. 모델 훈련 설정
print("3. 모델 훈련 설정...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
num_epochs = int(os.getenv('NUM_EPOCHS', 5))

print(f"훈련 설정:")
print(f"- 에포크 수: {num_epochs}")
print(f"- 초기 학습률: {optimizer.param_groups[0]['lr']}")
print(f"- 손실 함수: {criterion.__class__.__name__}")
print(f"- 옵티마이저: {optimizer.__class__.__name__}")
print(f"- L2 정규화 (Weight Decay): {optimizer.param_groups[0].get('weight_decay', 'N/A')}")
print(f"- 학습률 스케줄러: {scheduler.__class__.__name__}\n")

# 4. 모델 훈련 (피팅)
print("4. 모델 훈련 시작...")

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    total_batches = len(train_loader)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 진행 상황 출력 (매 100번째 배치마다)
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"  배치 [{batch_idx + 1}/{total_batches}] 평균 손실: {avg_loss:.4f}")
    
    # 에포크 완료 후 학습률 업데이트
    scheduler.step()
    epoch_loss = running_loss / total_batches
    current_lr = optimizer.param_groups[0]['lr']
    print(f"에포크 [{epoch+1}/{num_epochs}] 완료 - 평균 손실: {epoch_loss:.4f}, 학습률: {current_lr:.6f}")

print("훈련 완료!\n")

# 5. 모델 평가
print("5. 모델 평가...")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 클래스별 정확도 계산
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    
    # 전체 정확도
    overall_accuracy = correct / total
    print(f"전체 테스트 정확도: {overall_accuracy:.4f} ({correct}/{total})")
    
    # 클래스별 정확도
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck']
    print("\n클래스별 정확도:")
    for i in range(10):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            print(f"  {classes[i]}: {accuracy:.4f} ({int(class_correct[i])}/{int(class_total[i])})")

print("\n=== 튜토리얼 완료 ===")
print("ResNet 모델을 성공적으로 구현하고 훈련했습니다!")
print("\n주요 학습 내용:")
print("✓ Residual Block의 구현과 Skip Connection의 중요성")
print("✓ SOLID 원칙을 적용한 확장 가능한 아키텍처 설계")
print("✓ BasicBlock과 Bottleneck의 차이점")
print("✓ ResNet의 다양한 변형 (ResNet-18, 34, 50, 101)")
print("✓ Vanishing Gradient 문제 해결 방법") 