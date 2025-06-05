#!/bin/bash

# 환경변수 설정
export UV_TORCH_BACKEND=auto

# 환경변수 확인
echo "UV_TORCH_BACKEND가 $UV_TORCH_BACKEND로 설정되었습니다."
echo ""

# PyTorch 메인 프로그램 실행
echo "PyTorch 프로그램을 실행합니다..."
uv run python main.py
