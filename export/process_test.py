import os

print("=== Python 자식 프로세스에서 실행 중 ===")
print(f"Python 프로세스 ID: {os.getpid()}")
print(f"부모 프로세스 ID: {os.getppid()}")
print()
print("환경변수 확인:")
print(f"PARENT_VAR (export로 설정): {os.getenv('PARENT_VAR', '없음')}")
print(f"LOCAL_VAR (export 없이 설정): {os.getenv('LOCAL_VAR', '없음')}")
print()
print("=== Python 프로세스 종료 ===") 