"""
ResNet 아키텍처 분석 및 시각화
===============================

이 스크립트는 ResNet의 아키텍처를 분석하고 Skip Connection의 효과를 시각화합니다.
실제 학습은 하지 않고 구조 분석만 수행합니다.

주요 분석 내용:
1. ResNet과 일반 CNN의 구조 비교
2. Skip Connection의 gradient flow 개선 효과
3. 다양한 ResNet 변형의 파라미터 수 비교
4. Residual Block의 구조 분석
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

print("=== ResNet 아키텍처 분석 ===\n")


# ResNet 클래스들 정의 (분석용)
class ResidualBlock(nn.Module, ABC):
    """잔차 블록의 추상 기본 클래스"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파 정의 (하위 클래스에서 구현)"""
        pass


class BasicBlock(ResidualBlock):
    """기본 잔차 블록 구현"""
    
    expansion: int = 1  # 출력 채널 확장 비율
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
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
        """순전파: 잔차 연결 수행"""
        # 주 경로
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 잔차 연결: F(x) + x
        out += self.shortcut(x)
        out = torch.relu(out)
        
        return out


class Bottleneck(ResidualBlock):
    """병목 잔차 블록 구현 (더 깊은 네트워크용)"""
    
    expansion: int = 4  # 출력 채널 확장 비율
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
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


class ResNet(nn.Module):
    """ResNet 모델 구현"""
    
    def __init__(self, block_type: type[ResidualBlock], num_blocks: List[int], 
                 num_classes: int = 10) -> None:
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
    
    def _make_layer(self, block_type: type[ResidualBlock], planes: int, 
                   num_blocks: int, stride: int) -> nn.Sequential:
        """ResNet 레이어 생성 (내부 메서드)"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block_type(self.in_planes, planes, stride))
            self.in_planes = planes * block_type.expansion
        
        return nn.Sequential(*layers)
    
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
    """다양한 ResNet 변형을 생성하는 팩토리 클래스"""
    
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


class SimpleConvNet(nn.Module):
    """비교를 위한 일반적인 CNN 구조"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """모델의 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_resnet_variants():
    """다양한 ResNet 변형의 파라미터 수 분석"""
    print("1. ResNet 변형별 파라미터 수 분석")
    print("=" * 50)
    
    models = {
        'ResNet-18': ResNetFactory.resnet18(),
        'ResNet-34': ResNetFactory.resnet34(),
        'ResNet-50': ResNetFactory.resnet50(),
        'ResNet-101': ResNetFactory.resnet101(),
        'Simple CNN': SimpleConvNet()
    }
    
    results = []
    for name, model in models.items():
        params = count_parameters(model)
        results.append((name, params))
        print(f"{name:12s}: {params:,} 파라미터")
    
    print(f"\n💡 관찰:")
    print(f"   - ResNet-50부터는 Bottleneck 블록을 사용하여 효율성 증대")
    print(f"   - Skip Connection으로 깊은 네트워크도 안정적 훈련 가능")
    print(f"   - 일반 CNN 대비 ResNet은 더 깊으면서도 효율적\n")
    
    return results


def visualize_skip_connection():
    """Skip Connection의 효과 시각화"""
    print("2. Skip Connection의 Gradient Flow 효과")
    print("=" * 50)
    
    # 시뮬레이션된 gradient 값들
    layers = list(range(1, 19))  # 18개 레이어
    
    # 일반 CNN의 gradient (vanishing 현상)
    normal_gradients = [1.0 * (0.8 ** (18 - i)) for i in layers]
    
    # ResNet의 gradient (skip connection으로 안정화)
    resnet_gradients = [max(0.3, 1.0 * (0.95 ** (18 - i))) for i in layers]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(layers, normal_gradients, 'r-o', label='일반 CNN', linewidth=2)
    plt.plot(layers, resnet_gradients, 'b-s', label='ResNet', linewidth=2)
    plt.xlabel('레이어 깊이')
    plt.ylabel('Gradient 크기')
    plt.title('Gradient Flow 비교')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    # Skip connection 구조 시각화
    x = np.linspace(0, 4, 100)
    
    # Main path (비선형)
    main_path = np.sin(x) * 0.5 + 0.5
    
    # Skip connection (직선)
    skip_path = np.linspace(0.2, 0.8, 100)
    
    # 합성 경로
    combined = main_path + skip_path
    
    plt.plot(x, main_path, 'r--', label='Main Path F(x)', linewidth=2)
    plt.plot(x, skip_path, 'g-', label='Skip Connection x', linewidth=2)
    plt.plot(x, combined, 'b-', label='Output F(x) + x', linewidth=3)
    plt.xlabel('입력')
    plt.ylabel('출력')
    plt.title('Skip Connection 구조')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('explain/resnet_analysis.png', dpi=300, bbox_inches='tight')
    
    print("그래프가 'explain/resnet_analysis.png'로 저장되었습니다.")
    print(f"\n💡 Skip Connection의 효과:")
    print(f"   - Gradient vanishing 문제 완화")
    print(f"   - Identity mapping을 통한 안정적인 학습")
    print(f"   - 더 깊은 네트워크에서도 효과적인 역전파\n")


def analyze_block_structure():
    """Residual Block 구조 분석"""
    print("3. Residual Block 구조 분석")
    print("=" * 50)
    
    # BasicBlock 분석
    basic_block = BasicBlock(64, 64)
    basic_params = count_parameters(basic_block)
    
    # Bottleneck 분석
    bottleneck = Bottleneck(256, 64)
    bottleneck_params = count_parameters(bottleneck)
    
    print(f"BasicBlock (ResNet-18/34):")
    print(f"  - 구조: 3x3 conv → BN → ReLU → 3x3 conv → BN → (+skip) → ReLU")
    print(f"  - 파라미터 수: {basic_params:,}")
    print(f"  - 확장 비율: {BasicBlock.expansion}x")
    print(f"  - 사용 모델: ResNet-18, ResNet-34")
    
    print(f"\nBottleneck (ResNet-50+):")
    print(f"  - 구조: 1x1 conv → BN → ReLU → 3x3 conv → BN → ReLU → 1x1 conv → BN → (+skip) → ReLU")
    print(f"  - 파라미터 수: {bottleneck_params:,}")
    print(f"  - 확장 비율: {Bottleneck.expansion}x")
    print(f"  - 사용 모델: ResNet-50, ResNet-101, ResNet-152")
    
    print(f"\n💡 Bottleneck의 장점:")
    print(f"   - 1x1 conv로 차원 축소 후 3x3 conv 적용하여 계산량 감소")
    print(f"   - 더 깊은 네트워크를 효율적으로 구성 가능")
    print(f"   - 표현력은 유지하면서 파라미터 수는 절약\n")


def demonstrate_forward_pass():
    """순전파 과정 시연 (실제 계산 없이 구조만)"""
    print("4. ResNet 순전파 과정 시연")
    print("=" * 50)
    
    model = ResNetFactory.resnet18()
    model.eval()
    
    # 예시 입력 크기 (실제 데이터는 사용하지 않음)
    input_shape = (1, 3, 32, 32)
    
    print(f"입력 크기: {input_shape}")
    print(f"초기 conv 후: (1, 64, 32, 32)")
    print(f"Layer 1 후: (1, 64, 32, 32)")
    print(f"Layer 2 후: (1, 128, 16, 16)")
    print(f"Layer 3 후: (1, 256, 8, 8)")
    print(f"Layer 4 후: (1, 512, 4, 4)")
    print(f"Global Avg Pool 후: (1, 512, 1, 1)")
    print(f"최종 출력: (1, 10)")
    
    print(f"\n💡 특징:")
    print(f"   - 각 layer마다 채널 수는 2배씩 증가")
    print(f"   - 공간 해상도는 절반씩 감소")
    print(f"   - Skip connection으로 gradient flow 안정화")
    print(f"   - Global Average Pooling으로 위치 불변성 확보\n")


def compare_computational_complexity():
    """계산 복잡도 비교"""
    print("5. 계산 복잡도 비교")
    print("=" * 50)
    
    models = {
        'SimpleConv': SimpleConvNet(),
        'ResNet-18': ResNetFactory.resnet18(),
        'ResNet-50': ResNetFactory.resnet50()
    }
    
    for name, model in models.items():
        params = count_parameters(model)
        
        print(f"{name}:")
        print(f"  - 파라미터 수: {params:,}")
        print(f"  - 메모리 사용량: ~{params * 4 / (1024**2):.1f} MB")
        
    print(f"\n💡 효율성:")
    print(f"   - ResNet은 깊이 대비 파라미터 효율적")
    print(f"   - Skip connection으로 훈련 안정성 확보")
    print(f"   - Bottleneck 구조로 계산량 최적화\n")


if __name__ == "__main__":
    # 분석 실행 (학습은 하지 않음)
    analyze_resnet_variants()
    analyze_block_structure()
    demonstrate_forward_pass()
    compare_computational_complexity()
    visualize_skip_connection()
    
    print("=== 분석 완료 ===")
    print("ResNet의 핵심 아이디어와 구조적 장점을 확인했습니다!")
    print("실제 학습을 하려면 'python learn/06_resnet.py'를 실행하세요.") 