# learn/00_pytorch_core_components.py

# PyTorch 기본 구성 요소 (Core Components)

# 이 파일은 PyTorch의 핵심 개념 학습을 위한 코드 예제를 포함합니다.
# 예: 텐서, 자동 미분, 모듈 기본, 활성화 함수 등

import torch
import numpy as np # NumPy bridge 예제를 위해 import
import torch.nn as nn # nn.Module, nn.Sequential 등을 위해 import
import torch.nn.functional as F # 활성화 함수, 손실 함수 등을 위해 import
import torch.optim as optim # 옵티마이저를 위해 import
from torch.optim.lr_scheduler import StepLR # 학습률 스케줄러를 위해 import

# 1. 텐서 (Tensors)
print("--- 1. Tensors ---")
# 데이터로부터 직접 텐서 생성
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from data:\n {x_data}")

# NumPy 배열로부터 텐서 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from NumPy array:\n {x_np}")

# NumPy 배열 변경 시 텐서도 변경 (메모리 공유)
np_array[0, 0] = 10
print(f"NumPy array after change:\n {np_array}")
print(f"Tensor from NumPy array after change:\n {x_np}") # x_np도 변경됨
x_np[0, 0] = 1 # 텐서 변경 시 NumPy 배열도 변경
print(f"NumPy array after tensor change:\n {np_array}")

# PyTorch 텐서를 NumPy 배열로 변환
tensor_to_np = torch.ones(5)
numpy_from_tensor = tensor_to_np.numpy()
print(f"NumPy array from tensor:\n {numpy_from_tensor}")
tensor_to_np.add_(1) # 텐서 변경
print(f"NumPy array after original tensor changed:\n {numpy_from_tensor}") # NumPy 배열도 변경됨

# 다른 텐서로부터 텐서 생성 (구조는 유지, 값은 랜덤)
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")
x_zeros = torch.zeros_like(x_data)
print(f"Zeros Tensor: \n {x_zeros} \n")
x_rand_like = torch.rand_like(x_data, dtype=torch.float) # x_data의 자료형을 float으로 덮어씁니다.
print(f"Random Tensor (like x_data): \n {x_rand_like} \n")

# 모양(shape)을 사용한 텐서 생성
shape = (2, 3,)
rand_tensor = torch.rand(shape)
randn_tensor = torch.randn(shape) # 정규 분포
empty_tensor = torch.empty(shape) # 초기화되지 않은 값
arange_tensor = torch.arange(0, 10, 2) # 0부터 10 전까지 2 간격 (0, 2, 4, 6, 8)
linspace_tensor = torch.linspace(0, 10, 5) # 0부터 10까지 5개 균등 간격

print(f"Random Tensor with shape {shape}: \n {rand_tensor} \n")
print(f"Randn Tensor with shape {shape}: \n {randn_tensor} \n")
print(f"Empty Tensor with shape {shape}: \n {empty_tensor} \n") # 실행 시마다 다른 값
print(f"Arange Tensor: {arange_tensor}\n")
print(f"Linspace Tensor: {linspace_tensor}\n")

# 텐서 속성
tensor_attr = torch.rand(3,4)
print(f"Shape of tensor_attr: {tensor_attr.shape}")
print(f"Size of tensor_attr: {tensor_attr.size()}") # shape과 동일
print(f"Datatype of tensor_attr: {tensor_attr.dtype}")
print(f"Device tensor_attr is stored on: {tensor_attr.device}")
print(f"Number of dimensions of tensor_attr (ndim): {tensor_attr.ndim}")
print(f"Number of dimensions of tensor_attr (dim()): {tensor_attr.dim()}")

# 텐서 연산
# GPU가 존재하면 텐서를 이동합니다
if torch.cuda.is_available():
    tensor_op = tensor_attr.to('cuda')
    print(f"Device tensor_op is stored on after to('cuda'): {tensor_op.device}")
elif torch.backends.mps.is_available():
    tensor_op = tensor_attr.to('mps')
    print(f"Device tensor_op is stored on after to('mps'): {tensor_op.device}")
else:
    tensor_op = tensor_attr # CPU 그대로 사용
    print(f"Device tensor_op is stored on: {tensor_op.device}")


# NumPy 스타일의 인덱싱과 슬라이싱
indexing_tensor = torch.ones(4, 4)
print(f"First row: {indexing_tensor[0]}")
print(f"First column: {indexing_tensor[:, 0]}")
print(f"Last column: {indexing_tensor[..., -1]}")
indexing_tensor[:,1] = 0
print(f"Tensor after modifying one column:\n {indexing_tensor}")

# 텐서 합치기
t_cat = torch.cat([indexing_tensor, indexing_tensor, indexing_tensor], dim=1)
print(f"Concatenated tensor (dim=1):\n {t_cat}")
t_stack = torch.stack([indexing_tensor, indexing_tensor, indexing_tensor], dim=0) # 새로운 차원(dim=0)으로 쌓음
print(f"Stacked tensor (dim=0):\n {t_stack}, shape: {t_stack.shape}")

# 산술 연산
# 두 텐서 간의 행렬 곱(matrix multiplication)
y1 = indexing_tensor @ indexing_tensor.T
y2 = indexing_tensor.matmul(indexing_tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(indexing_tensor, indexing_tensor.T, out=y3)
print(f"Matrix multiplication (y1):\n {y1}")
# ... (기존 y2, y3 print 생략 가능, y1으로 대표)

# 요소별 곱(element-wise product)
z1 = indexing_tensor * indexing_tensor
z2 = indexing_tensor.mul(indexing_tensor)
z3 = torch.rand_like(indexing_tensor)
torch.mul(indexing_tensor, indexing_tensor, out=z3)
print(f"Element-wise product (z1):\n {z1}")
# ... (기존 z2, z3 print 생략 가능, z1으로 대표)

# 단일-요소 텐서
agg = indexing_tensor.sum()
agg_item = agg.item()
print(f"Sum of tensor: {agg}, Type: {type(agg)}")
print(f"Value of sum (item()): {agg_item}, Type: {type(agg_item)}")

# 바꿔치기(in-place) 연산
# 연산 결과를 피연산자에 저장하는 연산 (예: x.copy_(y), x.t_())은 _ 접미사를 갖습니다.
inplace_tensor = torch.ones(2,2)
print(f"Original tensor for inplace op:\n {inplace_tensor} \n")
inplace_tensor.add_(5)
print(f"Tensor after add_(5):\n {inplace_tensor}")

# 2. 자동 미분 (torch.autograd)
print("\n--- 2. Autograd ---")
x_ag = torch.ones(5)  # input tensor
y_ag = torch.zeros(3)  # expected output
w_ag = torch.randn(5, 3, requires_grad=True)
b_ag = torch.randn(3, requires_grad=True)
z_ag = torch.matmul(x_ag, w_ag)+b_ag
loss_ag = F.binary_cross_entropy_with_logits(z_ag, y_ag)

print(f"Gradient function for z_ag = {z_ag.grad_fn}")
print(f"Gradient function for loss_ag = {loss_ag.grad_fn}")

loss_ag.backward() # 역전파 실행
print(f"Gradient for w_ag:\n {w_ag.grad}")
print(f"Gradient for b_ag:\n {b_ag.grad}")

# 기울기 누적 확인
# w_ag.grad.zero_() # 보통 루프 시작 시점에 수행
# b_ag.grad.zero_()
# loss_ag.backward() # 두 번째 backward는 retain_graph=True 없이 오류 발생 (기본적으로 그래프 해제)
# print(f"Gradient for w_ag after second backward (w/o retain_graph):\n {w_ag.grad}")

# Gradient 추적 멈추기
# 1. torch.no_grad() 컨텍스트 매니저
with torch.no_grad():
    z_with_no_grad_block = torch.matmul(x_ag, w_ag)+b_ag
print(f"z_with_no_grad_block.requires_grad: {z_with_no_grad_block.requires_grad}")

# 2. .detach()
z_det = z_ag.detach() # z_ag와 데이터는 공유하지만, 연산 기록은 분리
print(f"z_det.requires_grad: {z_det.requires_grad}")

# 3. requires_grad_() (in-place)
temp_tensor = torch.randn(2,2, requires_grad=True)
print(f"temp_tensor.requires_grad before: {temp_tensor.requires_grad}")
temp_tensor.requires_grad_(False)
print(f"temp_tensor.requires_grad after: {temp_tensor.requires_grad}")


# 3. 모듈 및 클래스 기본 (torch.nn.Module 소개)
print("\n--- 3. nn.Module ---")
# 간단한 nn.Module 예시 (다음 섹션 "PyTorch 핵심 구현 패턴"에서 더 자세히 다룸)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x_input):
        x_output = self.linear1(x_input)
        x_output = self.relu(x_output)
        x_output = self.linear2(x_output)
        return x_output

model_example = SimpleModel()
print("SimpleModel instance:")
print(model_example)

# 모델 파라미터 확인
for name, param in model_example.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}, Shape: {param.shape}")

input_tensor_sm = torch.randn(1, 10)
output_tensor_sm = model_example(input_tensor_sm)
print(f"Input tensor shape for SimpleModel: {input_tensor_sm.shape}")
print(f"Output tensor shape for SimpleModel: {output_tensor_sm.shape}")

# nn.Sequential 사용 예시
seq_model = nn.Sequential(
    nn.Linear(10, 8),
    nn.ReLU(),
    nn.Linear(8, 5),
    nn.Tanh(),
    nn.Linear(5, 1)
)
print("\nnn.Sequential model instance:")
print(seq_model)
output_tensor_seq = seq_model(input_tensor_sm)
print(f"Output tensor shape for Sequential model: {output_tensor_seq.shape}")


# 4. 활성화 함수 (Activation Functions)
print("\n--- 4. Activation Functions ---")
input_activation = torch.randn(2, 3) * 2 # 다양한 값을 가지도록 조정
print(f"Input for Activations:\n {input_activation}")

relu_fn = nn.ReLU()
output_activation_relu = relu_fn(input_activation)
print(f"Output of ReLU:\n {output_activation_relu}")

sigmoid_fn = nn.Sigmoid()
output_activation_sigmoid = sigmoid_fn(input_activation)
print(f"Output of Sigmoid:\n {output_activation_sigmoid}")

tanh_fn = nn.Tanh()
output_activation_tanh = tanh_fn(input_activation)
print(f"Output of Tanh:\n {output_activation_tanh}")

leaky_relu_fn = nn.LeakyReLU(negative_slope=0.1)
output_activation_leaky = leaky_relu_fn(input_activation)
print(f"Output of LeakyReLU (negative_slope=0.1):\n {output_activation_leaky}")

# Softmax는 보통 다중 클래스 분류의 출력에 사용 (합이 1이 되는 확률 분포)
softmax_fn = nn.Softmax(dim=1) # dim=1은 각 행에 대해 softmax 적용
input_softmax = torch.randn(2, 4) # (batch_size, num_classes)
output_softmax = softmax_fn(input_softmax)
print(f"Input for Softmax:\n {input_softmax}")
print(f"Output of Softmax (dim=1):\n {output_softmax}")
print(f"Sum of Softmax output per row: {output_softmax.sum(dim=1)}")

# 5. 손실 함수 (Loss Functions)
print("\n--- 5. Loss Functions ---")
# 예시: MSE Loss (Mean Squared Error) - 회귀 문제
mse_loss_fn = nn.MSELoss()
predicted_mse = torch.randn(5, requires_grad=True)
target_mse = torch.randn(5)
loss_val_mse = mse_loss_fn(predicted_mse, target_mse)
print(f"Predicted (MSE): {predicted_mse.detach()}") # 출력용으로 detach
print(f"Target (MSE): {target_mse}")
print(f"MSE Loss: {loss_val_mse.item()}") # .item()으로 스칼라 값 추출
# loss_val_mse.backward() # 필요시 기울기 계산

# 예시: CrossEntropyLoss - 다중 클래스 분류 문제
# (내부적으로 LogSoftmax와 NLLLoss를 결합)
ce_loss_fn = nn.CrossEntropyLoss()
predicted_logits_ce = torch.randn(3, 5, requires_grad=True) # (batch_size, num_classes) - raw logits
target_labels_ce = torch.tensor([1, 0, 4]) # (batch_size) - class indices (0 ~ C-1)
loss_val_ce = ce_loss_fn(predicted_logits_ce, target_labels_ce)
print(f"Predicted Logits (CE, shape {predicted_logits_ce.shape}): \n {predicted_logits_ce.detach()}")
print(f"Target Labels (CE, shape {target_labels_ce.shape}): {target_labels_ce}")
print(f"CrossEntropy Loss: {loss_val_ce.item()}")

# 예시: BCEWithLogitsLoss - 이진 분류 또는 다중 레이블 분류 문제
# (Sigmoid + BCELoss 결합, 수치적 안정성)
bce_logits_loss_fn = nn.BCEWithLogitsLoss()
predicted_logits_bce = torch.randn(4, 1, requires_grad=True) # 4개 샘플, 이진 분류 (출력 로짓)
target_bce = torch.tensor([[1.0], [0.0], [1.0], [0.0]]) # 각 샘플의 타겟 (0 또는 1)
loss_val_bce_logits = bce_logits_loss_fn(predicted_logits_bce, target_bce)
print(f"Predicted Logits (BCEWithLogits): \n {predicted_logits_bce.detach()}")
print(f"Target (BCEWithLogits): \n {target_bce}")
print(f"BCEWithLogits Loss: {loss_val_bce_logits.item()}")


# 6. 옵티마이저 (torch.optim)
print("\n--- 6. Optimizers ---")
# 간단한 모델과 더미 데이터 생성
model_for_optim = nn.Linear(10, 1) # 입력 특성 10개, 출력 특성 1개
dummy_input = torch.randn(5, 10) # 배치 크기 5, 입력 특성 10개
dummy_target = torch.randn(5, 1)  # 배치 크기 5, 출력 특성 1개

# 옵티마이저 정의 (예: SGD)
optimizer_sgd = optim.SGD(model_for_optim.parameters(), lr=0.01, momentum=0.9)

# 옵티마이저 정의 (예: Adam)
optimizer_adam = optim.Adam(model_for_optim.parameters(), lr=0.001)

# 학습률 스케줄러 예시 (StepLR)
# 2 에폭마다 학습률에 gamma(0.1)를 곱함
scheduler_steplr = StepLR(optimizer_adam, step_size=2, gamma=0.1)

print(f"Initial LR for Adam: {optimizer_adam.param_groups[0]['lr']}")

# 일반적인 훈련 스텝 (간략화된 루프)
loss_fn_optim = nn.MSELoss()
for epoch in range(5): # 5 에폭만 예시로 실행
    # 1. 옵티마이저의 기울기 초기화
    optimizer_adam.zero_grad() # Adam 옵티마이저 사용 예시

    # 2. 모델을 통해 예측 수행
    predictions = model_for_optim(dummy_input)

    # 3. 손실 계산
    loss_optim = loss_fn_optim(predictions, dummy_target)

    # 4. 역전파 수행
    loss_optim.backward()

    # 5. 옵티마이저 스텝 (파라미터 업데이트)
    optimizer_adam.step()
    
    # 6. 스케줄러 스텝 (에폭 단위)
    scheduler_steplr.step()

    print(f"Epoch {epoch+1}, Loss: {loss_optim.item():.4f}, Current LR: {optimizer_adam.param_groups[0]['lr']:.5f}")

print("Optimizer step and LR scheduler example completed.")


# 7. 장치 할당 (.to(device))
print("\n--- 7. Device Allocation ---")
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using {device} device')

# 모델과 텐서를 해당 장치로 옮김
tensor_to_move = torch.randn(3,3)
print(f"Tensor original device: {tensor_to_move.device}")
tensor_to_move_on_device = tensor_to_move.to(device) # 새로운 변수에 할당
print(f"Tensor new device: {tensor_to_move_on_device.device}")

model_to_move = SimpleModel() # 위에서 정의한 SimpleModel 사용
print(f"Model original device (any parameter): {next(model_to_move.parameters()).device}")
model_to_move.to(device) # 모델 자체를 이동 (in-place 효과)
print(f"Model new device (any parameter): {next(model_to_move.parameters()).device}")

# 장치 간 데이터 이동 시 주의: 연산은 동일 장치에 있는 텐서들 간에만 가능
try:
    cpu_tensor = torch.randn(3,3, device='cpu')
    # device가 cpu가 아닌 경우에만 gpu_tensor를 다른 장치로 만듦
    if device != 'cpu':
        gpu_tensor = torch.randn(3,3).to(device)
        result = cpu_tensor + gpu_tensor # 오류 발생 예상
        print(f"Result of cpu + gpu tensor (should not happen): {result}")
    else:
        print("Skipping cross-device operation test as device is CPU.")
except RuntimeError as e:
    print(f"Expected error for cross-device operation: {e}")


# 8. 가중치 초기화 (Weight Initialization)
print("\n--- 8. Weight Initialization ---")
# nn.Module의 가중치는 기본적으로 특정 방식으로 초기화됨 (예: Linear, Conv2d는 Kaiming He 초기화 등)
linear_layer_default_init = nn.Linear(5, 2)
print(f"Default initialized weights for Linear layer:\n {linear_layer_default_init.weight.data}") # .data로 값만 출력
print(f"Default initialized bias for Linear layer:\n {linear_layer_default_init.bias.data}")

# 수동으로 가중치 초기화 예시
linear_layer_manual_init = nn.Linear(5, 2)

# Xavier Uniform 초기화
nn.init.xavier_uniform_(linear_layer_manual_init.weight)
# Bias는 0으로 초기화
nn.init.zeros_(linear_layer_manual_init.bias)

print(f"Manually initialized weights (Xavier Uniform):\n {linear_layer_manual_init.weight.data}")
print(f"Manually initialized bias (Zeros):\n {linear_layer_manual_init.bias.data}")

# 모델 전체에 적용하는 함수 예시
class ModelWithCustomInit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # 예시 Conv2d 추가
        self.bn1 = nn.BatchNorm2d(16) # 예시 BatchNorm2d 추가
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(16 * 8 * 8, 128) # 입력 크기는 conv, bn 통과 후 flatten 가정 (예시 크기)
        self.fc2 = nn.Linear(128, 10)
        self._initialize_weights() # 생성자에서 호출

    def forward(self, x_input): # (batch, 3, H, W) 형태의 입력 가정 (예: 3x8x8)
        x_output = self.conv1(x_input)
        x_output = self.bn1(x_output)
        x_output = self.relu1(x_output)
        x_output = x_output.view(x_output.size(0), -1) # Flatten
        x_output = self.fc1(x_output)
        x_output = self.fc2(x_output)
        return x_output

    @torch.no_grad() # 초기화 과정은 기울기 추적 불필요
    def _initialize_weights(self):
        print("Applying custom initialization...")
        for m in self.modules(): # self.modules()는 자기 자신 포함 모든 sub-module을 순회
            if isinstance(m, nn.Conv2d):
                print(f"Initializing Conv2d: {m}")
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                print(f"Initializing BatchNorm2d: {m}")
                nn.init.constant_(m.weight, 1) # 보통 weight는 1
                nn.init.constant_(m.bias, 0)   # bias는 0
            elif isinstance(m, nn.Linear):
                print(f"Initializing Linear: {m}")
                nn.init.xavier_normal_(m.weight) # Linear는 Xavier Normal 예시
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# 모델 인스턴스 생성 시 _initialize_weights 호출됨
model_custom_init = ModelWithCustomInit()
print("ModelWithCustomInit instantiated and weights initialized.")
# 예시로 fc1 레이어 가중치 일부 확인
print(f"Layer fc1 weights after custom init (first 5 values): {model_custom_init.fc1.weight.data.flatten()[:5]}")
print(f"Layer conv1 weights after custom init (first 5 values of first filter): {model_custom_init.conv1.weight.data[0,0,0,:5].flatten()}")


print("\nPyTorch Core Components 스크립트 실행 완료.") 