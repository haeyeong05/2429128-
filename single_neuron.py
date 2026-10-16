"""
뉴런 1개의 계산 과정 (PPT 3번째 장: 인공신경망의 기본 틀)
Inputs -> Weights -> Sum -> + Bias -> Activation Function -> Output
필요 라이브러리: numpy
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 1) Inputs: 모델에 넣는 값 (예: 정규화한 키, 허리)
x = np.array([1.0, 2.0])

# 2) Weights: 각 입력의 중요도
w = np.array([0.5, 0.3])

# 3) Bias: 입력과 상관없이 더하는 기본값
b = 0.1

# 4) Sum (Σ): 입력 x 가중치를 모두 더함
weighted = x * w
total = weighted.sum()
print(f"Inputs            x = {x}")
print(f"Weights           w = {w}")
print(f"입력 x 가중치       = {weighted}")
print(f"Sum (Σ w·x)         = {total:.2f}")
print(f"Sum + Bias          = {total + b:.2f}")

# 5) Activation Function: 합계를 0~1 사이 값으로 변환
z = total + b
y = sigmoid(z)

# 6) Output
print(f"Activation sigmoid  = {y:.4f}")
print(f"Output y            = {y:.4f}")

# 참고: bias가 출력을 어떻게 옮기는지 확인 (입력이 모두 0이어도 출력이 생김)
x0 = np.array([0.0, 0.0])
print(f"\n[입력이 전부 0일 때] bias=0.1 -> y = {sigmoid(x0 @ w + 0.1):.4f}, "
      f"bias=0 -> y = {sigmoid(x0 @ w + 0.0):.4f}")
