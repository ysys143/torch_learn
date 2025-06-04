#!/bin/bash

echo "=== 자식 프로세스에서 실행 중 ==="
echo "자식 프로세스 ID: $$"
echo "부모 프로세스 ID: $PPID"
echo ""
echo "환경변수 확인:"
echo "PARENT_VAR (export로 설정): $PARENT_VAR"
echo "LOCAL_VAR (export 없이 설정): $LOCAL_VAR"
echo ""
echo "=== 자식 프로세스 종료 ===" 