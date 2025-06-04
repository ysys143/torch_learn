#!/bin/bash

echo "=== env_test.sh 스크립트 시작 ==="

# 환경변수 설정
export SCRIPT_VAR="스크립트에서 설정된 값"
TEST_VAR="export 없이 설정된 값"

echo "스크립트 내부에서:"
echo "SCRIPT_VAR = $SCRIPT_VAR"
echo "TEST_VAR = $TEST_VAR"

echo "=== env_test.sh 스크립트 끝 ===" 