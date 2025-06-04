# PyTorch 딥러닝 학습 로드맵

이 문서는 PyTorch를 사용하여 딥러닝의 다양한 아키텍처와 개념을 학습하기 위한 로드맵을 제공합니다.

## 학습 목표

- 딥러닝의 기본적인 원리 이해
- 다양한 신경망 아키텍처의 구현 및 활용 능력 습득
- 실제 데이터를 이용한 모델 훈련, 평가, 추론 과정 숙달
- PyTorch 프레임워크의 효율적인 사용 방법 익히기

## 학습 순서 및 내용

개념적 연관성과 난이도를 고려하여 다음 순서로 학습을 진행합니다.

### 1. 기초 다지기

1.  **로지스틱 회귀 (Logistic Regression)**
    -   **내용**: 이진 분류를 위한 가장 기본적인 모델. 선형 결합 후 시그모이드 함수 적용.
    -   **주요 학습**: 데이터 로드 및 전처리, PyTorch 모델 정의, 손실 함수 및 옵티마이저 설정, 훈련 루프, 평가 및 추론.

2.  **FLC (Fully Connected Layer) / 다층 퍼셉트론 (Multi-Layer Perceptron, MLP)**
    -   **내용**: 여러 개의 완전 연결층을 가진 신경망. 딥러닝의 가장 기본적인 형태.
    -   **주요 학습**: 은닉층, 활성화 함수 (ReLU, Sigmoid 등), 역전파의 개념, 다중 분류/회귀.

3.  **Autoencoder (오토인코더)**
    -   **내용**: 데이터를 압축(인코딩)하고 다시 재구성(디코딩)하는 비지도 학습 모델.
    -   **주요 학습**: 차원 축소, 특성 학습, 노이즈 제거, 이상 탐지.

### 2. 이미지 데이터 처리

4.  **CNN (Convolutional Neural Network)**
    -   **내용**: 이미지와 같은 격자형 데이터 처리에 특화된 신경망. 컨볼루션 필터와 풀링 계층 사용.
    -   **주요 학습**: 컨볼루션 연산, 풀링, 특성 맵, 이미지 분류.

5.  **VGG Network**
    -   **내용**: 깊은 신경망 구조의 중요성을 보여준 초기 CNN 아키텍처. 작은 필터와 깊은 층으로 구성.
    -   **주요 학습**: 네트워크 깊이의 영향, 단순한 구조로 복잡한 문제 해결.

6.  **ResNet (Residual Network)**
    -   **내용**: 잔차 연결(Residual Connection)을 도입하여 깊은 신경망의 훈련을 용이하게 한 아키텍처.
    -   **주요 학습**: Skip connection, vanishing gradient 문제 해결, 매우 깊은 네트워크 훈련.

7.  **U-Net**
    -   **내용**: 이미지 분할(Segmentation)에 특화된 아키텍처. 인코더-디코더 구조와 스킵 연결 활용.
    -   **주요 학습**: 이미지 세그멘테이션, 의료 영상 분석 응용.

### 3. 텍스트 및 순차 데이터 처리

8.  **Word2Vec & GloVe**
    -   **내용**: 단어를 벡터 공간에 임베딩하는 초기 모델. 단어의 의미적 유사성을 벡터로 표현.
    -   **주요 학습**: 단어 임베딩의 개념, CBOW 및 Skip-gram 모델, 텍스트의 수치화.

9.  **RNN (Recurrent Neural Network)**
    -   **내용**: 순차 데이터를 처리하는 신경망. 이전 스텝의 정보를 현재 스텝에 전달하는 순환 구조.
    -   **주요 학습**: 시퀀스 데이터, 은닉 상태, vanishing/exploding gradient 문제.

10. **LSTM (Long Short-Term Memory) & GRU (Gated Recurrent Unit)**
    -   **내용**: RNN의 장기 의존성 문제(Long-Term Dependency Problem)를 해결하기 위한 게이트 메커니즘을 포함한 변형.
    -   **주요 학습**: 셀 상태, 게이트 (입력, 망각, 출력), GRU의 간소화된 구조, 더 긴 시퀀스 처리.

11. **Bi-LSTM (Bidirectional LSTM)**
    -   **내용**: 양방향 정보를 활용하여 현재 시점의 문맥을 더 잘 이해하는 LSTM 변형.
    -   **주요 학습**: 과거 및 미래 정보 통합, 문맥 이해도 향상.

12. **Seq2Seq (Sequence-to-Sequence)**
    -   **내용**: 인코더-디코더 구조를 기반으로 한 시퀀스 변환 모델.
    -   **주요 학습**: 인코더-디코더 아키텍처, 문맥 벡터, 기계 번역, 요약.

13. **Attention Mechanism (어텐션 메커니즘)**
    -   **내용**: Seq2Seq 모델의 성능을 향상시키기 위해 도입된 개념. 입력 시퀀스 중 중요한 부분에 집중.
    -   **주요 학습**: 가중치 부여, 정보 집중, Transformer의 기초.

14. **ELMo (Embeddings from Language Models)**
    -   **내용**: 사전 훈련된 언어 모델로부터 문맥에 따라 단어 임베딩을 생성하는 모델.
    -   **주요 학습**: 문맥적 단어 임베딩, 다의어 처리, 현대 NLP의 발전.

### 4. 고급 개념 (심화 학습)

#### 4.1 고급 NLP 모델

15. **BERT (Bidirectional Encoder Representations from Transformers)**
    -   **내용**: 양방향 문맥을 학습하는 혁신적인 언어 모델. Masked Language Model (MLM)과 Next Sentence Prediction (NSP)을 통해 사전 학습.
    -   **주요 학습**: 트랜스포머 인코더, 사전 학습 및 파인튜닝, 문맥적 임베딩, 다양한 NLP 태스크 적용.

16. **Sentence-BERT (SBERT)**
    -   **내용**: BERT를 확장하여 의미론적으로 유사한 문장 임베딩을 효율적으로 생성하는 모델. 문장 유사도, 의미 검색 등에 활용.
    -   **주요 학습**: Siamese 및 Triplet 네트워크 구조, 의미론적 유사성 측정, 문장 임베딩의 활용.

17. **ColBERT (Contextualized Late Interaction over BERT)**
    -   **내용**: BERT의 문맥적 임베딩을 활용하면서도 검색 효율성을 높이기 위해 지연 상호작용(Late Interaction)을 도입한 검색 모델.
    -   **주요 학습**: 효율적인 랭킹, Query 및 Document 임베딩, 정보 검색 시스템.

#### 4.2 기타 고급 개념

18. **Siamese Network (샴 네트워크)**
    -   **내용**: 두 개 이상의 입력에 대해 유사도 학습을 수행하는 신경망.
    -   **주요 학습**: 메트릭 학습, 얼굴 인식, 서명 검증.

19. **GAN (Generative Adversarial Network)**
    -   **내용**: 생성자(Generator)와 판별자(Discriminator)가 경쟁하며 학습하는 생성 모델.
    -   **주요 학습**: 생성 모델링, 게임 이론적 접근, 이미지 생성, 데이터 증강.

## 환경 설정

-   **Python**: 최신 버전 (예: 3.9+)
-   **PyTorch**: `uv pip install torch`를 사용하여 설치
-   **패키지 관리**: `uv` 사용 (`uv add [package_name]`) 

이 로드맵은 제안 사항이며, 학습자의 필요와 관심사에 따라 유연하게 조정될 수 있습니다. 