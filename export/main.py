import os
import torch
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

def main():
    """PyTorch 환경 설정 및 기본 동작 확인"""
    
    print("=== PyTorch 환경 확인 ===")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
    
    print("\n=== 환경변수 확인 ===")
    uv_torch_backend = os.getenv('UV_TORCH_BACKEND')
    print(f"UV_TORCH_BACKEND: {uv_torch_backend}")
    
    print("\n=== 간단한 PyTorch 텐서 연산 ===")
    # 간단한 텐서 생성 및 연산
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.matmul(x, y)
    
    print("행렬 X:")
    print(x)
    print("\n행렬 Y:")
    print(y)
    print("\n행렬곱 결과 (X @ Y):")
    print(z)
    
    # 디바이스 정보
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n사용 중인 디바이스: {device}")
    
    # GPU가 있다면 GPU에서 연산 테스트
    if torch.cuda.is_available():
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        z_gpu = torch.matmul(x_gpu, y_gpu)
        print("GPU에서 동일한 연산 수행 완료")
    
    print("\n=== PyTorch 환경 설정 완료 ===")

if __name__ == "__main__":
    main()
