# PyTorch 학습 프로젝트

이 프로젝트는 PyTorch 학습을 위한 환경 설정과 export, source 명령어 이해를 위한 예시를 포함합니다.

## 환경변수 설정

### UV_TORCH_BACKEND 설정

프로젝트에서 `UV_TORCH_BACKEND=auto` 환경변수를 사용하기 위한 여러 방법이 구성되어 있습니다.

## export와 source 명령어 이해

### export 명령어

`export`는 환경변수를 설정하고 자식 프로세스에게 전달하는 명령어입니다.

#### 특징
- 현재 쉘과 모든 자식 프로세스에서 사용 가능
- 현재 쉘 세션에서만 유효 (터미널 종료 시 사라짐)
- export 없이 설정한 변수는 자식 프로세스에 전달되지 않음

#### 사용법
```bash
# 환경변수 설정
export MY_VAR="값"

# 환경변수 확인
echo $MY_VAR

# 모든 환경변수 확인
env
```

### 현재 쉘과 자식 프로세스의 관계

#### 프로세스 구조
```
현재 쉘 (zsh/bash)
├── python script.py        ← 자식 프로세스
├── ./script.sh            ← 자식 프로세스  
├── uv run python main.py   ← 자식 프로세스
└── 기타 실행하는 모든 프로그램들 ← 자식 프로세스들
```

#### export의 중요성

| 변수 설정 방법 | 현재 쉘에서 접근 | 자식 프로세스에서 접근 |
|---------------|-----------------|---------------------|
| `export VAR="값"` | ✅ 가능 | ✅ 가능 |
| `VAR="값"` (export 없음) | ✅ 가능 | ❌ 불가능 |

#### 실제 테스트 예시

현재 쉘에서 변수 설정:
```bash
export PARENT_VAR="부모쉘에서 설정"    # export로 설정
LOCAL_VAR="부모쉘에서만 설정"          # export 없이 설정
```

자식 프로세스에서 확인:
```bash
# Bash 스크립트에서
echo "PARENT_VAR: $PARENT_VAR"  # 출력: 부모쉘에서 설정
echo "LOCAL_VAR: $LOCAL_VAR"    # 출력: (빈 값)
```

```python
# Python 스크립트에서
import os
print(os.getenv('PARENT_VAR'))  # 출력: 부모쉘에서 설정  
print(os.getenv('LOCAL_VAR'))   # 출력: None
```

#### 프로세스 ID 확인

현재 쉘과 자식 프로세스의 관계를 확인하는 방법:

```bash
# 현재 쉘 프로세스 ID
echo $$

# 자식 프로세스에서 부모 프로세스 ID 확인
echo $PPID
```

#### 왜 이것이 중요한가?

- **Python 프로그램**에서 `os.getenv()`로 읽을 수 있는 것은 `export`된 환경변수만
- **UV_TORCH_BACKEND=auto**를 Python에서 사용하려면 반드시 `export`가 필요
- **자식 프로세스**는 부모 쉘의 `export`된 환경변수만 상속받음

### source 명령어

`source`는 스크립트 파일을 현재 쉘에서 실행하여 환경변수나 함수를 현재 세션에 적용하는 명령어입니다.

#### 특징
- 서브쉘을 생성하지 않음
- 스크립트의 모든 변수와 함수가 현재 쉘에 적용
- 설정 파일이나 환경변수 파일 로드에 주로 사용

#### 사용법
```bash
# 방법 1: source 명령어 사용
source .env

# 방법 2: 점(.) 명령어 사용 (동일한 기능)
. .env

# 스크립트 파일 로드
source ./env_test.sh
```

### 실행 방식의 차이

| 실행 방법 | 설명 | 변수 유지 |
|----------|------|----------|
| `./script.sh` | 서브쉘에서 실행 | ❌ 유지되지 않음 |
| `source script.sh` | 현재 쉘에서 실행 | ✅ 유지됨 |

## 프로젝트별 환경변수 설정 방법

### 1. .env 파일 사용 (권장)

`.env` 파일에 환경변수를 저장하고 Python에서 `python-dotenv`로 로드:

```bash
# .env 파일 내용
UV_TORCH_BACKEND=auto
```

```python
# Python 코드에서 사용
import os
from dotenv import load_dotenv

load_dotenv()
torch_backend = os.getenv('UV_TORCH_BACKEND')
```

### 2. 실행 스크립트 사용

`run.sh` 스크립트로 환경변수 설정 후 실행:

```bash
#!/bin/bash
export UV_TORCH_BACKEND=auto
echo "UV_TORCH_BACKEND가 $UV_TORCH_BACKEND로 설정되었습니다."
# 실제 프로젝트 실행 명령어 추가
```

실행 방법:
```bash
chmod +x run.sh
./run.sh
```

### 3. direnv 사용

`.envrc` 파일로 디렉토리 진입 시 자동 로드:

```bash
# .envrc 파일 내용
export UV_TORCH_BACKEND=auto
```

direnv 설치 후 사용:
```bash
# direnv 허용
direnv allow
```

### 4. 수동 로드

필요할 때마다 수동으로 환경변수 로드:

```bash
# .env 파일 로드
source .env

# 환경변수 확인
echo $UV_TORCH_BACKEND
```

## 실습 예시

### export와 source 차이점 확인

`env_test.sh` 스크립트를 통해 차이점을 확인할 수 있습니다:

```bash
# 1. 일반 실행 (서브쉘)
./env_test.sh
echo $SCRIPT_VAR  # 빈 값 출력

# 2. source 실행 (현재 쉘)
source env_test.sh
echo $SCRIPT_VAR  # 설정된 값 출력
```

### 프로세스와 환경변수 관계 테스트

현재 쉘과 자식 프로세스 간의 환경변수 전달을 확인:

```bash
# 1. 환경변수 설정
export PARENT_VAR="부모쉘에서 설정"    # export로 설정
LOCAL_VAR="부모쉘에서만 설정"          # export 없이 설정

# 2. 현재 쉘에서 확인
echo "PARENT_VAR: $PARENT_VAR"
echo "LOCAL_VAR: $LOCAL_VAR"

# 3. Bash 자식 프로세스에서 테스트
./process_test.sh

# 4. Python 자식 프로세스에서 테스트  
python process_test.py
```

예상 결과:
- 현재 쉘: 두 변수 모두 접근 가능
- 자식 프로세스: `PARENT_VAR`만 접근 가능, `LOCAL_VAR`는 접근 불가

## 패키지 관리

이 프로젝트는 `uv`를 사용하여 패키지를 관리합니다:

```bash
# 패키지 추가
uv add python-dotenv

# 가상환경 활성화
source .venv/bin/activate

# 프로젝트 실행
uv run python example.py
```

## 파일 구조

```
pytorch_learn/
├── .env                 # 환경변수 설정
├── .envrc              # direnv용 환경변수
├── run.sh              # 실행 스크립트
├── env_test.sh         # export/source 테스트 스크립트
├── process_test.sh     # 프로세스와 환경변수 테스트 스크립트 (bash)
├── process_test.py     # 프로세스와 환경변수 테스트 스크립트 (python)
├── example.py          # 환경변수 로드 예시
├── main.py             # 메인 프로그램
├── pyproject.toml      # 프로젝트 설정
├── uv.lock            # 의존성 잠금 파일
└── README.md          # 이 파일
```

## 주요 명령어 정리

```bash
# 환경변수 설정
export UV_TORCH_BACKEND=auto

# .env 파일 로드
source .env

# 프로젝트 실행
uv run python main.py

# 환경변수 확인
echo $UV_TORCH_BACKEND
env | grep UV_TORCH_BACKEND
```
