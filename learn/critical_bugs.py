# learn/critical_bugs.py
# 심각한 보안 및 성능 버그들

import os
import sqlite3
import hashlib

# 실제 패턴의 시크릿들 (더 realistic하게)
AWS_ACCESS_KEY = "AKIA1234567890ABCDEF"  # AWS 패턴
AWS_SECRET_KEY = "abcdefghijklmnop1234567890ABCDEFGHIJKLMN"
GITHUB_TOKEN = "ghp_1234567890abcdefghijklmnopqrstuvwxyz123"
OPENAI_API_KEY = "sk-proj-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# SQL Injection 취약점
def get_user_data(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # 매우 위험한 SQL injection
    query = f"SELECT * FROM users WHERE id = {user_id}"  # 정수 injection
    cursor.execute(query)
    
    # 더 위험한 문자열 injection
    def get_user_by_name(username):
        query = f"SELECT * FROM users WHERE name = '{username}'"
        cursor.execute(query)
        return cursor.fetchall()
    
    return cursor.fetchall()

# 명백한 보안 문제
def unsafe_file_operations():
    # 경로 조작 취약점
    def read_file(filename):
        # ../../../etc/passwd 같은 경로 조작 가능
        with open(f"/app/files/{filename}", 'r') as f:
            return f.read()
    
    # 커맨드 인젝션
    def process_file(filename):
        os.system(f"cat {filename}")  # 매우 위험
        os.popen(f"ls -la {filename}").read()  # 역시 위험

# 메모리 누수와 성능 문제
class MemoryLeak:
    def __init__(self):
        self.data = []
    
    def add_data(self, item):
        # 메모리가 계속 증가
        self.data.append(item)
        # 정리하지 않음
    
    def process_large_data(self):
        # 엄청난 메모리 사용
        huge_list = []
        for i in range(10**7):  # 1천만 개
            huge_list.append([0] * 1000)  # 각각 1000개 원소
        return huge_list

# 무한 루프
def infinite_loop():
    counter = 0
    while counter >= 0:  # 버그: counter가 계속 증가
        counter += 1
        print(f"Counter: {counter}")
        # 종료 조건 없음

# 잘못된 암호화
def bad_crypto():
    # 약한 해시 알고리즘
    password = "user_password"
    weak_hash = hashlib.md5(password.encode()).hexdigest()  # MD5는 취약
    
    # 솔트 없는 해시
    def hash_password(pwd):
        return hashlib.sha1(pwd.encode()).hexdigest()  # 솔트 없음

# 예외 처리 문제
def bad_exception_handling():
    try:
        # 위험한 작업들
        result = 10 / 0
        file = open("nonexistent.txt")
        data = undefined_variable  # NameError
    except:  # 모든 예외를 잡음
        pass  # 무시함 - 매우 나쁜 패턴

# 레이스 컨디션
import threading
import time

shared_counter = 0

def unsafe_increment():
    global shared_counter
    for _ in range(100000):
        # 원자적이지 않은 연산
        temp = shared_counter
        time.sleep(0.000001)  # 경쟁 상황 유발
        shared_counter = temp + 1

def create_race_condition():
    threads = []
    for i in range(5):
        t = threading.Thread(target=unsafe_increment)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

# 잘못된 입력 검증
def no_input_validation(user_input):
    # 입력 검증 없음
    eval(user_input)  # 매우 위험! 코드 실행 가능
    exec(user_input)  # 더 위험!
    
    # 파일 시스템 접근 검증 없음
    with open(user_input, 'w') as f:
        f.write("data")

# 하드코딩된 설정들
DATABASE_URL = "postgresql://admin:password123@localhost:5432/proddb"
SECRET_KEY = "super-secret-key-dont-change"
DEBUG = True  # 프로덕션에서 디버그 모드

# 잘못된 로깅
import logging

def bad_logging():
    # 민감한 정보 로깅
    username = "admin"
    password = "secret123"
    
    logging.info(f"User login: {username} with password: {password}")  # 비밀번호 로깅!
    
    # 너무 많은 로깅
    for i in range(100000):
        logging.info(f"Processing item {i}")  # 성능 문제

# 타입 안정성 문제
def type_confusion():
    # 타입 체크 없음
    def add_numbers(a, b):
        return a + b  # 문자열이 들어올 수 있음
    
    # 잘못된 타입 변환
    user_age = input("나이를 입력하세요: ")
    if user_age > 18:  # 문자열과 숫자 비교
        print("성인입니다")

# 위험한 직렬화
import pickle

def unsafe_deserialization(data):
    # pickle은 임의 코드 실행 가능
    return pickle.loads(data)  # 매우 위험

# 버퍼 오버플로우 가능성 (Python에서는 드물지만)
def potential_overflow():
    # 엄청난 크기의 문자열
    huge_string = "A" * (10**9)  # 1GB 문자열
    return huge_string

if __name__ == "__main__":
    print("심각한 버그들이 포함된 코드 실행 중...")
    print(f"AWS Key: {AWS_ACCESS_KEY}")  # 시크릿 노출
    print(f"GitHub Token: {GITHUB_TOKEN}")  # 토큰 노출
    
    # SQL Injection 시연
    malicious_input = "1 OR 1=1; DROP TABLE users; --"
    get_user_data(malicious_input)
    
    # 기타 위험한 함수들
    no_input_validation("__import__('os').system('rm -rf /')")
    unsafe_deserialization(b"dangerous_data")
    
    print("모든 버그 시연 완료!") 