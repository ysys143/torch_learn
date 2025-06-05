import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# 환경변수 확인
torch_backend = os.getenv('UV_TORCH_BACKEND')
print(f"UV_TORCH_BACKEND: {torch_backend}")

# 여기에 실제 PyTorch 코드를 작성
