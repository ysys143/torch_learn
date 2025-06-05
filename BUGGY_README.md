# PyTorch 버그가 많은 예제 프로젝트

이 프로젝트는 의도적으로 버그를 포함하고 있습니다!!!

## 설치 방법

```bash
pip install pytorch  # 잘못된 패키지명
pip install torchvision
pip instal numpy  # 오타: install
```

## 사용 방법

1. 파일을 다운로드합니다
2. 실행합니다:

```python
python buggy_pytorch_example.py  # 이 파일은 실행되지 않을 것입니다!
```

### 주의사항

- 이 코드는 **매우 위험**합니다
- 절대로 프로덕션에서 사용하지 마세요!!
- 보안 취약점이 있습니다...

## 기능 목록

* [ ] 기본 모델 구현
* [x] 버그 있는 모델 구현
* [ ] 테스트 코드 (하지만 테스트는 없습니다)
* [x] 보안 취약점들
* [ ] 문서화 (이 README도 엉망입니다)

## 링크들

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [깨진 링크](https://this-link-does-not-exist.com/404)
- [또 다른 깨진 링크](https://broken.link)
- [로컬 파일 링크](./non_existent_file.md)

## 코드 예시

이것은 작동하지 않는 코드입니다:

```python
# 잘못된 문법
class Model(nn.Module):
    def __init__(self)
        # 콜론 누락
        pass
```

더 많은 버그:

```python
import torch

# 이것은 오류가 발생합니다
result = 1 + "문자열"
print(result)
```

## 이미지

![존재하지 않는 이미지](./images/missing_image.png)
![또 다른 깨진 이미지](https://broken-image-url.com/image.jpg)

## 표 (형식이 잘못됨)

| 컬럼1 | 컬럼2 컬럼3 |  # 표 형식 오류
|-------|-------------|
| 데이터1 | 데이터2 | 데이터3 |  # 컬럼 수 불일치
| 데이터4 데이터5 |  # 파이프 누락

## 잘못된 제목 계층

# 제목 1
### 제목 3 (제목 2를 건너뜀)
##### 제목 5 (제목 4를 건너뜀)

## HTML 태그 남용

<div style="color: red;">이것은 마크다운에서 권장되지 않습니다</div>
<script>alert('XSS 취약점')</script>

## 일관성 없는 스타일

- 불릿 포인트
* 다른 불릿 포인트
+ 또 다른 불릿 포인트

1. 번호 목록
3. 잘못된 번호
1. 다시 1번

## 잘못된 코드 블록

```
# 언어 지정 없는 코드 블록
def broken_function():
    return "이것은 문법 강조가 안 됩니다"

## 마크다운 문법 오류들

**볼드체가 닫히지 않음

*이탤릭체도 닫히지 않음

[링크 텍스트](#anchor-that-does-not-exist)

## 메타데이터 문제

<!-- TODO: 이 문서를 완전히 다시 작성해야 함 -->
<!-- FIXME: 모든 링크가 깨져있음 -->
<!-- HACK: 임시 해결책으로 작성됨 -->

---

**경고**: 이 프로젝트는 교육 목적으로만 사용되어야 하며, 실제 코드에서는 이런 패턴들을 피해야 합니다!

버전: 0.0.1-alpha-buggy
작성자: 버그봇
날짜: 2025년 13월 32일  <!-- 불가능한 날짜 -->

## 연락처

이메일: admin@[도메인없음]
GitHub: [사용자명없음]
웹사이트: http://unsecure-website.com  <!-- HTTP 사용 -->

---

마지막 업데이트: ??/??/???? 