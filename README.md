# PyTorch 딥러닝 학습 로드맵

이 문서는 PyTorch를 사용하여 딥러닝의 다양한 아키텍처와 개념을 학습하기 위한 로드맵을 제공합니다.

## 학습 목표

- 딥러닝의 기본적인 원리 이해
- 다양한 신경망 아키텍처의 구현 및 활용 능력 습득
    - 실제 데이터를 이용한 모델 훈련, 평가, 추론 과정 숙달
- PyTorch 프레임워크의 효율적인 사용 방법 익히기

## 학습 순서 및 내용

개념적 연관성과 난이도를 고려하여 다음 순서로 학습을 진행합니다.

### 0. PyTorch 기초 (PyTorch Fundamentals)

0.  **PyTorch 기본 구성 요소 (Core Components)**
    -   **내용**: PyTorch 라이브러리의 기본적인 구조와 핵심 컴포넌트에 대한 이해를 목표로 합니다. 텐서(Tensor)의 개념, 다양한 자료형, 데이터 표현 방식, 기본적인 연산 및 계산 그래프(Computational Graph)의 작동 원리를 학습합니다.
    -   **주요 학습**:
        -   **텐서 (Tensors)**: PyTorch의 기본 데이터 구조. NumPy 배열과 유사하지만 GPU 가속 지원. 다양한 차원, 자료형(float, int, bool 등) 학습.
        -   **자동 미분 (`torch.autograd`)**: 역전파 알고리즘을 위한 미분 자동 계산 기능. `requires_grad`, `.backward()`.
        -   **모듈 및 클래스 기본 (`torch.nn.Module` 소개)**: 신경망 레이어나 모델 전체를 구성하는 기본 블록. `__init__`을 통한 레이어 정의, `forward`를 통한 데이터 흐름 정의.
        -   **활성화 함수 (Activation Functions)**: `torch.nn.ReLU`, `torch.nn.Sigmoid`, `torch.nn.Tanh` 등 다양한 활성화 함수의 종류와 역할.
        -   **손실 함수 (Loss Functions)**: `torch.nn.CrossEntropyLoss`, `torch.nn.MSELoss` 등 모델의 예측과 실제 값 간의 차이를 측정하는 함수.
        -   **옵티마이저 (`torch.optim`)**: `torch.optim.Adam`, `torch.optim.SGD` 등 손실 함수를 기반으로 모델 파라미터를 업데이트하는 알고리즘.
        -   **장치 할당 (`.to(device)`)**: 모델과 텐서를 CPU 또는 GPU (예: CUDA, MPS)로 옮기는 방법.
        -   **가중치 초기화 (Weight Initialization)**: 모델 성능에 영향을 미치는 파라미터 초기화 방법 (`torch.nn.init`).

1.  **PyTorch 핵심 구현 패턴 (Key Implementation Patterns)** ✅
    -   **내용**: 실제 모델 구현에 필수적인 PyTorch의 주요 클래스 상속 방식, 데이터 처리, 모델 상태 관리 및 훈련/평가 제어 심층 학습. "반복 훈련"을 통해 체득하는 것을 목표로 합니다.
    -   **주요 학습**:
        -   `torch.nn.Module` 상속 활용: `__init__`에서의 레이어 선언, `forward` 메소드 설계 및 데이터 흐름 구성 심층 분석.
        -   커스텀 데이터셋 구축: `torch.utils.data.Dataset` 및 `torch.utils.data.DataLoader`를 활용한 다양한 데이터 로딩 및 전처리 파이프라인 구축.
        -   훈련/평가 모드 제어: `model.train()`, `model.eval()`의 정확한 역할 및 차이 (예: Dropout, BatchNorm), `torch.no_grad()` 컨텍스트 매니저의 효과적인 사용법 및 비교 실험.
        -   모델 상태 저장 및 로드: `model.state_dict()`를 이용한 방식과 전체 모델(`torch.save(model, PATH)`)을 저장하는 방식의 차이점, 장단점, 사용 시나리오 명확히 구분.
        -   훈련 루프 상세 구성: 데이터 로딩, 모델 예측, 손실 계산, 역전파, 옵티마이저 스텝, 평가 지표 계산 등 완전한 훈련 사이클 구현.
        -   Hooks 활용 (선택 사항): `register_forward_hook`, `register_backward_hook` 등을 사용한 중간 피처/그래디언트 추출 및 분석 기초.
    -   **구현 파일**: 
        -   `learn/01_pytorch_implementation_patterns.py` - 실행 가능한 구현 코드
        -   `explain/01_pytorch_implementation_patterns_explanation.md` - 상세한 이론적 설명
        -   `notebook/01_pytorch_implementation_patterns.ipynb` - 인터랙티브 노트북

### 1. 기초 다지기

2.  **로지스틱 회귀 (Logistic Regression)**
    -   **내용**: 이진 분류를 위한 가장 기본적인 모델. 선형 결합 후 시그모이드 함수 적용.
    -   **주요 학습**: 데이터 로드 및 전처리, PyTorch 모델 정의, 손실 함수 및 옵티마이저 설정, 훈련 루프, 평가 및 추론.

3.  **FLC (Fully Connected Layer) / 다층 퍼셉트론 (Multi-Layer Perceptron, MLP)**
    -   **내용**: 여러 개의 완전 연결층을 가진 신경망. 딥러닝의 가장 기본적인 형태.
    -   **주요 학습**: 은닉층, 활성화 함수 (ReLU, Sigmoid 등), 역전파의 개념, 다중 분류/회귀.
        -   아키텍처 유연성 확보: 동일 MLP 구조를 다양한 스타일로 반복 구현 (Functional API 방식, `nn.Sequential` 활용, `torch.nn.Module` 클래스 상속 방식).

4.  **Autoencoder (오토인코더)**
    -   **내용**: 데이터를 압축(인코딩)하고 다시 재구성(디코딩)하는 비지도 학습 모델.
    -   **주요 학습**: 차원 축소, 특성 학습, 노이즈 제거, 이상 탐지.

### 2. 이미지 데이터 처리

5.  **CNN (Convolutional Neural Network)**
    -   **내용**: 이미지와 같은 격자형 데이터 처리에 특화된 신경망. 컨볼루션 필터와 풀링 계층 사용.
    -   **주요 학습**: 컨볼루션 연산, 풀링, 특성 맵, 이미지 분류.
        -   구현 다양성 탐구: 기본 CNN 모델을 여러 스타일 (Functional, `nn.Sequential`, 클래스 상속)로 작성하며 각 방식의 장단점 이해.

6.  **VGG Network**
    -   **내용**: 깊은 신경망 구조의 중요성을 보여준 초기 CNN 아키텍처. 작은 필터와 깊은 층으로 구성.
    -   **주요 학습**: 네트워크 깊이의 영향, 단순한 구조로 복잡한 문제 해결.

7.  **ResNet (Residual Network)**
    -   **내용**: 잔차 연결(Residual Connection)을 도입하여 깊은 신경망의 훈련을 용이하게 한 아키텍처.
    -   **주요 학습**: Skip connection, vanishing gradient 문제 해결, 매우 깊은 네트워크 훈련.
        -   (심화 과제) 논문 원본(예: "Deep Residual Learning for Image Recognition")을 참고하여 그림 없이 PyTorch 코드로 직접 구현 시도 및 공식 구현과 비교 분석.

8.  **U-Net**
    -   **내용**: 이미지 분할(Segmentation)에 특화된 아키텍처. 인코더-디코더 구조와 스킵 연결 활용.
    -   **주요 학습**: 이미지 세그멘테이션, 의료 영상 분석 응용.

9.  **Siamese Network (샴 네트워크)**
    -   **내용**: 두 개 이상의 입력에 대해 유사도 학습을 수행하는 신경망. 이미지 유사도 측정에 특화.
    -   **주요 학습**: 메트릭 학습, 얼굴 인식, 서명 검증, 이미지 유사도 측정, 원샷 학습.

### 3. 텍스트 및 순차 데이터 처리

10. **Word2Vec & GloVe**
    -   **내용**: 단어를 벡터 공간에 임베딩하는 초기 모델. 단어의 의미적 유사성을 벡터로 표현.
    -   **주요 학습**: 단어 임베딩의 개념, CBOW 및 Skip-gram 모델, 텍스트의 수치화.

11. **RNN (Recurrent Neural Network)**
    -   **내용**: 순차 데이터를 처리하는 신경망. 이전 스텝의 정보를 현재 스텝에 전달하는 순환 구조.
    -   **주요 학습**: 시퀀스 데이터, 은닉 상태, vanishing/exploding gradient 문제.

12. **LSTM (Long Short-Term Memory) & GRU (Gated Recurrent Unit)**
    -   **내용**: RNN의 장기 의존성 문제(Long-Term Dependency Problem)를 해결하기 위한 게이트 메커니즘을 포함한 변형.
    -   **주요 학습**: 셀 상태, 게이트 (입력, 망각, 출력), GRU의 간소화된 구조, 더 긴 시퀀스 처리.

13. **Bi-LSTM (Bidirectional LSTM)**
    -   **내용**: 양방향 정보를 활용하여 현재 시점의 문맥을 더 잘 이해하는 LSTM 변형.
    -   **주요 학습**: 과거 및 미래 정보 통합, 문맥 이해도 향상.

14. **Encoder-Decoder Architecture (인코더-디코더 아키텍처)**
    -   **내용**: 입력 데이터를 압축된 중간 표현으로 변환하는 인코더와 이를 출력으로 복원하는 디코더로 구성된 일반적 프레임워크.
    -   **주요 학습**: 정보 압축 및 복원, 표현 학습, 다양한 도메인 간 변환의 기본 원리.

15. **Seq2Seq (Sequence-to-Sequence)**
    -   **내용**: 인코더-디코더 구조를 기반으로 한 시퀀스 변환 모델.
    -   **주요 학습**: 인코더-디코더 아키텍처 실제 적용, 문맥 벡터, 기계 번역, 요약.

16. **Attention Mechanism (어텐션 메커니즘)**
    -   **내용**: Seq2Seq 모델의 성능을 향상시키기 위해 도입된 개념. 입력 시퀀스 중 중요한 부분에 집중.
    -   **주요 학습**: 가중치 부여, 정보 집중, Transformer의 기초.

### 4. Hugging Face Transformers 시작하기

17. **Hugging Face Transformers 라이브러리 이해**
    -   **내용**: Hugging Face Transformers 라이브러리의 핵심 아키텍처, PyTorch 모델과의 연동 방식(모델 래핑), 모델 허브(Model Hub)를 통한 사전 훈련된 모델 및 토크나이저의 검색, 로드(`from_pretrained`), 저장(`save_pretrained`) 메커니즘, 관련 주요 파일(`config.json`, `pytorch_model.bin`, `tokenizer.json` 등)의 역할과 구조, `AutoModel`, `AutoTokenizer`, `pipeline` API 등 핵심 컴포넌트 활용법, 그리고 `datasets`, `evaluate` 라이브러리와의 연동 등 Hugging Face 생태계 전반을 학습합니다.
    -   **주요 학습**: `transformers` 라이브러리 설치 및 기본 설정, 모델 허브 탐색 및 활용, PyTorch 모델을 Hugging Face 프레임워크와 통합하는 방법, 토크나이저의 중요성과 다양한 토크나이저 활용, 간단한 NLP 태스크에 `pipeline` 적용, 사전 훈련된 모델 기반의 전이 학습 준비.

### 5. Transformer 기반 모델

18. **Transformer**
    -   **내용**: Self-Attention 메커니즘을 기반으로 한 혁신적인 아키텍처. RNN 없이도 순차 데이터 처리 가능.
    -   **주요 학습**: Self-Attention, Multi-Head Attention, Position Encoding, 병렬 처리의 장점.

19. **BERT (Bidirectional Encoder Representations from Transformers)**
    -   **내용**: 양방향 문맥을 학습하는 사전 훈련된 Transformer 기반 언어 모델.
    -   **주요 학습**: 사전 훈련 및 파인튜닝, Masked Language Model, 양방향 문맥 이해.
        -   (심화 과제) Hugging Face `transformers` 라이브러리의 `BertModel` 내부 모듈(예: `BertEmbeddings`, `BertLayer`, `BertPooler`) 구조를 코드를 통해 분석하고, 각 컴포넌트의 역할과 데이터 흐름 이해.

20. **GPT-2 (Generative Pre-trained Transformer 2)**
    -   **내용**: 자기회귀 방식의 언어 생성에 특화된 Transformer 기반 모델. 대규모 텍스트 데이터로 사전 훈련된 생성형 언어 모델.
    -   **주요 학습**: 언어 생성, 프롬프트 엔지니어링, 대화형 AI, 창작 및 요약, 제로샷 학습.

### 6. 생성 모델 (Generative Models)

21. **VAE (Variational Autoencoder)**
    -   **내용**: 변분 추론을 사용하여 확률적 인코더-디코더 구조를 학습하는 생성 모델. 잠재 공간에서 연속적인 표현 학습.
    -   **주요 학습**: 변분 추론, KL 발산, 재매개화 트릭, 잠재 공간 탐색, 확률적 생성.

22. **GAN (Generative Adversarial Network)**
    -   **내용**: 생성자(Generator)와 판별자(Discriminator)가 경쟁하며 학습하는 생성 모델.
    -   **주요 학습**: 생성 모델링, 게임 이론적 접근, 이미지 생성, 데이터 증강, 적대적 훈련, 모드 붕괴 문제.

23. **Diffusion Models**
    -   **내용**: 노이즈 제거 과정을 통해 데이터를 생성하는 확률적 생성 모델. 순방향 확산과 역방향 디노이징 과정으로 구성.
    -   **주요 학습**: 확산 과정, 디노이징, 스케줄링, DDPM/DDIM, 조건부 생성, 고품질 이미지 합성.

### 7. 강화 학습 (Reinforcement Learning)

24. **강화 학습 기초 (Reinforcement Learning Fundamentals)**
    -   **내용**: 에이전트가 환경과 상호작용하며 보상을 최대화하는 행동을 학습하는 인공지능 패러다임. 순차적 의사결정 문제 해결.
    -   **주요 학습**: 에이전트, 환경, 상태(State), 행동(Action), 보상(Reward), 정책(Policy), 가치 함수(Value Function), 마르코프 결정 과정(MDP), 탐험과 활용(Exploration vs. Exploitation).

25. **Q-Learning**
    -   **내용**: 가치 기반 강화 학습 알고리즘으로, Q-테이블 또는 Q-함수를 통해 각 상태-행동 쌍의 최적 가치를 학습.
    -   **주요 학습**: Q-테이블 업데이트, 벨만 방정식(Bellman Equation), 이산 행동 공간, 오프-정책(Off-policy) 학습.

26. **DQN (Deep Q-Networks)**
    -   **내용**: Q-Learning에 딥러닝(신경망)을 적용하여 복잡한 상태 공간에서 최적의 행동 가치 함수를 학습하는 알고리즘.
    -   **주요 학습**: 경험 리플레이(Experience Replay), 타겟 네트워크(Target Network), 밸류 기반 접근의 확장, 게임 플레이 응용.

27. **정책 경사법 (Policy Gradient Methods)**
    -   **내용**: 정책 자체를 직접 최적화하여 보상을 최대화하는 행동을 선택하는 알고리즘.
    -   **주요 학습**: 정책 함수, 확률적 정책, REINFORCE 알고리즘, 온-정책(On-policy) 학습, 행동 가치 함수(Advantage Function) 개념 도입.

28. **PPO (Proximal Policy Optimization)**
    -   **내용**: 정책 경사법의 불안정성을 개선하고, 신뢰 영역 내에서 정책 업데이트를 수행하여 안정적이고 효율적인 학습을 가능하게 하는 알고리즘. Actor-Critic 구조를 활용하는 경우가 많습니다.
    -   **주요 학습**: 클리핑된 목적 함수(Clipped Surrogate Objective), 신뢰 영역 방법(Trust Region Methods), Advantage 추정, GAE (Generalized Advantage Estimation), 샘플 효율성, 다양한 연속/이산 제어 문제에 대한 적용.

### 8. 고급 개념 (실용적 응용)

#### 8.1 임베딩 모델

29. **Sentence-BERT (SBERT)**
    -   **내용**: BERT를 확장하여 의미론적으로 유사한 문장 임베딩을 효율적으로 생성하는 모델.
    -   **주요 학습**: Siamese 및 Triplet 네트워크 구조, 의미론적 유사성 측정, 문장 임베딩의 활용.
        -   (심화 과제) SBERT 모델 (예: `sentence_transformers` 라이브러리 내 모델)의 `forward` 메소드 실행 시, 입력 텍스트가 임베딩으로 변환되는 전체 과정을 `print()`문, 디버거, 또는 `wandb.watch()`와 같은 도구를 활용하여 단계별로 추적 및 시각화.

30. **ColBERT (Contextualized Late Interaction over BERT)**
    -   **내용**: BERT의 문맥적 임베딩을 활용하면서도 검색 효율성을 높이기 위해 지연 상호작용을 도입한 검색 모델.
    -   **주요 학습**: 효율적인 랭킹, Query 및 Document 임베딩, 정보 검색 시스템.

31. **SimCSE**
    -   **내용**: Unsupervised 및 Supervised 방식을 모두 활용하여 문장 임베딩의 품질을 향상시키는 Contrastive Learning 기반의 방법론.
    -   **주요 학습**: Contrastive Learning, 데이터 증강 (텍스트의 경우 dropout을 노이즈로 활용), Unsupervised SimCSE (입력 문장을 두 번 인코딩하여 positive pair로 사용), Supervised SimCSE (NLI 데이터셋 등에서 entailment, contradiction 쌍 활용), 임베딩 공간의 정렬(alignment) 및 균일성(uniformity).

32. **CLIP (Contrastive Language-Image Pre-training)**
    -   **내용**: 텍스트와 이미지 간의 관계를 학습하여 두 모달리티를 연결하는 모델. 이미지에 대한 텍스트 설명, 텍스트에 대한 이미지 검색 등 다양한 멀티모달 작업에 활용.
    -   **주요 학습**: 대조 학습(Contrastive Learning), 이미지 인코더(ResNet 또는 Vision Transformer), 텍스트 인코더(Transformer), 제로샷 이미지 분류, 임베딩 공간의 의미론적 일치.

#### 8.2 Transformer 고급 기법

33. **RoPE (Rotary Positional Embedding)**
    -   **내용**: Transformer 모델에서 위치 정보를 인코딩하는 방법 중 하나로, 상대적 위치 정보를 효율적으로 반영.
    -   **주요 학습**: 상대적 위치 임베딩, 컨텍스트 길이 확장, 모델 일반화 성능 향상, 효율적인 병렬 처리.

34. **Sparse Attention**
    -   **내용**: Self-Attention의 계산 복잡도를 줄이기 위해 모든 토큰 간의 어텐션 대신 일부 토큰에만 집중하는 기법.
    -   **주요 학습**: 어텐션 효율성 개선, 긴 시퀀스 처리, 메모리 및 연산량 절감, 다양한 희소 어텐션 패턴 (예: Longformer, BigBird).

35. **FlashAttention**
    -   **내용**: 긴 시퀀스에 대한 어텐션 계산 시 I/O 효율성을 최적화하여 GPU 메모리 사용량을 줄이고 속도를 크게 향상시킨 기법.
    -   **주요 학습**: Tiling 및 recomputation 기법, 정확한 어텐션 계산, 하드웨어 특성 고려, LLM 학습 및 추론 가속화.

36. **GQA (Grouped-Query Attention) / MQA (Multi-Query Attention)**
    -   **내용**: Multi-Head Attention에서 여러 쿼리 헤드가 단일 또는 그룹화된 키(key) 및 값(value) 프로젝션을 공유하여 추론 시 메모리 대역폭 요구량을 줄이고 처리 속도를 높이는 기법.
    -   **주요 학습**: 추론 최적화, 메모리 효율성, LLM 서빙, 다양한 모델 아키텍처(예: Llama)에서의 활용.

37. **Mixture of Experts (MoE)**
    -   **내용**: 모델의 전체 파라미터 수를 늘리면서도 각 입력에 대해 일부 \"전문가\" 네트워크만 활성화하여 계산 효율성을 유지하는 기법. 대규모 모델의 학습 및 추론에 효과적.
    -   **주요 학습**: 라우팅 메커니즘, 전문가 네트워크, 로드 밸런싱, 모델 용량 확장, LLM의 스케일링.

#### 8.3 연속 시스템 모델링 (Continuous System Modeling)

38. **Neural ODE (Ordinary Differential Equations)**
    -   **내용**: 연속적인 시간 동역학을 모델링하는 신경망. 미분방정식 해법기를 사용하여 연속적인 깊이의 네트워크 구현.
    -   **주요 학습**: 연속적 네트워크 구조, ODE 솔버, 메모리 효율적 역전파, 물리학 기반 모델링, 시계열 예측.

#### 8.4 Transformer 기반 시각 모델

39. **Vision Transformer (ViT)**
    -   **내용**: 이미지를 패치로 분할한 후 Transformer 모델을 적용하여 이미지 분류를 수행하는 모델. CNN 없이도 이미지 특징 학습 가능.
    -   **주요 학습**: 이미지 패치 임베딩, Positional Embedding, Self-Attention을 이용한 이미지 처리, 대규모 데이터셋 사전 훈련의 중요성.

40. **Swin Transformer**
    -   **내용**: 계층적 Transformer 구조와 Shifted Window Attention을 도입하여 다양한 스케일의 시각 특징을 효율적으로 학습하는 모델. ViT의 계산 복잡도 및 성능 문제를 개선.
    -   **주요 학습**: 계층적 특징 표현, Shifted Window Attention, Swin Block, 이미지 분류 및 dense prediction 태스크 (예: 객체 탐지, 분할) 적용.

#### 8.5 LLM 파인튜닝 기법 (LLM Fine-tuning Techniques)

41. **SFT (Supervised Fine-tuning)**
    -   **내용**: 사전 훈련된 언어 모델(PLM)을 특정 작업에 맞춰 지도 학습 방식으로 전체 또는 일부를 재훈련하는 기법.
    -   **주요 학습**: PLM의 전이 학습(Transfer Learning), 특정 도메인/태스크 적응, 데이터셋 구축 및 관리, 오버피팅 방지 전략.

42. **PEFT (Parameter-Efficient Fine-tuning)**
    -   **내용**: PLM의 모든 파라미터를 파인튜닝하는 대신, 적은 수의 파라미터만 조정하여 효율성을 높이는 기법.
    -   **주요 학습**: 전이 학습 효율성 증대, 학습 비용 절감, LoRA, Prompt Tuning, Prefix Tuning 등 다양한 PEFT 방법론.

43. **LoRA (Low-Rank Adaptation)**
    -   **내용**: PEFT의 한 종류로, 사전 훈련된 모델의 가중치 행렬에 작은 저랭크 행렬을 추가하여 효율적으로 파인튜닝하는 기법.
    -   **주요 학습**: 저랭크 분해, 가중치 업데이트의 효율성, 모델 저장 공간 절약, 다양한 모델에 LoRA 적용.

44. **RLHF (Reinforcement Learning from Human Feedback)**
    -   **내용**: 인간의 피드백을 활용하여 언어 모델의 응답을 개선하는 강화 학습 기반 파인튜닝 기법. 정렬(Alignment) 문제 해결에 핵심.
    -   **주요 학습**: 보상 모델(Reward Model), 정책 최적화(Policy Optimization), PPO(Proximal Policy Optimization), 인간 피드백의 중요성, LLM의 안전성 및 유용성 향상.

45. **DPO (Direct Preference Optimization)**
    -   **내용**: 강화 학습 기반 인간 피드백 학습(RLHF)을 단순화하여 선호도 데이터를 직접 최적화하는 방법. LLM의 정렬(Alignment)에 주로 사용.
    -   **주요 학습**: 인간 선호도 학습, RLHF의 대안, 보상 모델 없이 직접 최적화, LLM의 안전성 및 유용성 개선.

46. **스페셜 토큰을 통한 음성 처리 (Voice Processing with Special Tokens)**
    -   **내용**: 언어 모델의 토큰화 시스템에 스페셜 토큰을 도입하여 음성 관련 작업을 수행하는 방법.
    -   **주요 학습**: 음성 특징(예: Mel-spectrogram)의 토큰화, 음성-텍스트 변환(STT), 텍스트-음성 변환(TTS), 음성 관련 작업에 PLM 활용.

#### 8.6 추론/배포 최적화 (Inference/Deployment Optimization)

47. **Test-time Scaling**
    -   **내용**: 모델의 추론 단계에서 성능을 향상시키기 위해 다양한 스케일링 기법을 적용하는 전략.
    -   **주요 학습**: 데이터 증강(Data Augmentation)을 통한 여러 버전의 입력 생성, 앙상블(Ensemble) 기법, TTA(Test-Time Augmentation)의 개념 및 적용, 추론 시 안정성과 정확도 향상.

## 환경 설정

-   **Python**: 최신 버전 (예: 3.9+)
-   **PyTorch**: `uv pip install torch`를 사용하여 설치
-   **패키지 관리**: `uv` 사용 (`uv add [package_name]`) 

이 로드맵은 제안 사항이며, 학습자의 필요와 관심사에 따라 유연하게 조정될 수 있습니다. 
