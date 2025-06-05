# learn/problematic_config.py
# 문제가 있는 설정 파일 - 보안 및 성능 이슈 포함

import os
import subprocess

# 보안 버그 1: 하드코딩된 비밀번호
DATABASE_PASSWORD = "admin123"  # 보안 위험!
API_KEY = "sk-1234567890abcdef"  # API 키 노출!

# 보안 버그 2: SQL 인젝션 취약점
def unsafe_query(user_input):
    query = f"SELECT * FROM users WHERE name = '{user_input}'"  # SQL 인젝션 위험
    return query

# 보안 버그 3: 커맨드 인젝션 위험
def execute_command(filename):
    # 사용자 입력을 직접 shell 명령어에 사용
    os.system(f"cat {filename}")  # 위험한 코드
    subprocess.run(f"rm {filename}", shell=True)  # 더 위험한 코드

# 성능 버그 1: 비효율적인 반복문
def inefficient_processing():
    result = []
    for i in range(10000):
        for j in range(10000):
            # 불필요한 중첩 루프
            result.append(i * j)
    return result

# 성능 버그 2: 메모리 비효율
class MemoryWaster:
    def __init__(self):
        # 불필요하게 큰 데이터 구조
        self.huge_list = [i for i in range(1000000)]
        self.duplicate_data = self.huge_list.copy()  # 중복 데이터

# 논리 버그: 잘못된 계산
def calculate_accuracy(correct, total):
    if total = 0:  # 버그: == 대신 =
        return 0
    return correct / total * 100

# 타입 버그: 잘못된 타입 사용
def process_numbers(numbers):
    # 문자열과 숫자 혼합
    result = 0
    for num in numbers:
        result = result + num + "error"  # 타입 오류
    return result

# 예외 처리 버그: 너무 넓은 except
def risky_operation():
    try:
        # 위험한 연산
        result = 10 / 0
        return result
    except:  # 모든 예외를 잡음 (나쁜 관례)
        pass  # 오류를 무시

# 전역 변수 남용
global_counter = 0

def modify_global():
    global global_counter
    global_counter += 1  # 전역 변수 수정

def another_modifier():
    global global_counter  
    global_counter *= 2  # 예측하기 어려운 부작용

# 잘못된 상수 정의
PI = 3.14  # 정확하지 않은 값
MAX_USERS = -1  # 논리적으로 말이 안 되는 값

# Deprecated 함수 사용 (경고가 있을 수 있음)
import warnings
warnings.filterwarnings("ignore")  # 모든 경고 무시 (나쁜 관례)

# 리소스 누수
def file_leak():
    f = open("somefile.txt", "w")
    f.write("data")
    # f.close() 누락 - 파일 핸들 누수

# 무한 재귀 가능성
def recursive_function(n):
    if n > 0:
        return recursive_function(n + 1)  # 버그: n이 계속 증가
    return n

# 잘못된 인코딩 처리
def encoding_problem():
    text = "한글 텍스트"
    # 인코딩 지정 없이 파일 쓰기
    with open("output.txt", "w") as f:
        f.write(text)  # 인코딩 문제 가능

# 하드코딩된 경로 (이식성 문제)
LOG_FILE = "/Users/specific_user/logs/app.log"  # 하드코딩된 경로
CONFIG_PATH = "C:\\Windows\\config.ini"  # 윈도우 전용 경로

# 잘못된 정규표현식
import re

def bad_regex(text):
    # 성능이 나쁜 정규표현식 (ReDoS 가능성)
    pattern = r"(a+)+$"
    return re.match(pattern, text)

# 동시성 문제 (race condition 가능성)
import threading

shared_resource = 0

def unsafe_increment():
    global shared_resource
    temp = shared_resource
    # 다른 스레드가 여기서 shared_resource를 변경할 수 있음
    shared_resource = temp + 1

# 메인 실행부에서 버그 있는 함수들 호출
if __name__ == "__main__":
    print("문제가 있는 설정 및 함수들을 실행합니다...")
    
    # 각종 버그 있는 함수들 호출
    unsafe_query("'; DROP TABLE users; --")
    execute_command("important_file.txt; rm -rf /")  # 매우 위험!
    
    print(f"DATABASE_PASSWORD: {DATABASE_PASSWORD}")  # 비밀번호 로그에 노출
    
    risky_operation()
    modify_global()
    another_modifier()
    
    print("실행 완료 (보안 및 성능 문제가 많습니다!)") 