"""
실습 6. 선형회귀의 한계 vs 신경망 (PPT '전통적 선형회귀분석의 한계: 비선형성')
입력 1개(x), 출력 y = sin(x) 형태의 비선형 관계에서 두 모델을 비교
필요 라이브러리: numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 비선형 데이터 생성
n = 300
x = np.random.uniform(-3, 3, n).reshape(-1, 1)
y = np.sin(1.5 * x) + np.random.normal(0, 0.1, (n, 1))

idx = np.random.permutation(n)
tr, te = idx[:240], idx[240:]
xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]

# 1) 선형회귀 (y = a x + b)
A = np.column_stack([xtr, np.ones(len(xtr))])
coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
lin_pred = np.column_stack([xte, np.ones(len(xte))]) @ coef

# 2) 신경망: 입력 1 -> 은닉 8(시그모이드) -> 출력 1(선형)
H = 8
W1 = np.random.randn(1, H) * 1.0
b1 = np.zeros((1, H))
W2 = np.random.randn(H, 1) * 0.5
b2 = np.zeros((1, 1))
lr = 0.05
for ep in range(8000):
    a1 = sigmoid(xtr @ W1 + b1)
    pred = a1 @ W2 + b2
    err = pred - ytr
    d_pred = 2 * err / len(xtr)
    dW2 = a1.T @ d_pred
    db2 = d_pred.sum(axis=0, keepdims=True)
    d_z1 = (d_pred @ W2.T) * a1 * (1 - a1)
    dW1 = xtr.T @ d_z1
    db1 = d_z1.sum(axis=0, keepdims=True)
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1


def nn_predict(xin):
    return sigmoid(xin @ W1 + b1) @ W2 + b2


nn_pred = nn_predict(xte)


def metrics(true, pred):
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)
    return rmse, r2


lin_rmse, lin_r2 = metrics(yte, lin_pred)
nn_rmse, nn_r2 = metrics(yte, nn_pred)
print("실습 6. 비선형 데이터 y = sin(1.5x) + noise  (검증 데이터 60개)")
print(f"선형회귀 : RMSE = {lin_rmse:.3f}, R^2 = {lin_r2:.3f}")
print(f"신경망   : RMSE = {nn_rmse:.3f}, R^2 = {nn_r2:.3f}")

xs = np.linspace(-3, 3, 300).reshape(-1, 1)
plt.figure(figsize=(7, 4))
plt.scatter(xtr, ytr, s=10, color="lightgray", label="train data")
plt.plot(xs, np.column_stack([xs, np.ones(len(xs))]) @ coef, color="tab:red", label=f"linear regression (R2={lin_r2:.2f})")
plt.plot(xs, nn_predict(xs), color="tab:blue", label=f"neural network (R2={nn_r2:.2f})")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear regression vs neural network (nonlinear data)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("nonlinear_comparison.png", dpi=130)
plt.close()
print("그래프 저장: nonlinear_comparison.png")
