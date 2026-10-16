"""
데이터처리개론 4. 인공신경망 기초 - 실습 코드
PPT 슬라이드 내용(knot->km/h 뉴런, 시그모이드, 3-3-1 신경망, 이미지 행렬)을 코드로 구현
필요 라이브러리: numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------------------------------------------------------
# 실습 1. 인공신경망의 느낌: knot -> km/h (뉴런 1개, 가중치 1개)
#   PPT의 km->mile 예제를 선박 속도 단위로 바꿈
#   km/h = knot x w,  정답 = 37.04 (20노트), 입력 = 20
#   (1노트 = 1.852 km/h 이므로 w가 1.852에 가까워지면 정답)
# ------------------------------------------------------------------
print("=" * 60)
print("실습 1. knot -> km/h 단일 뉴런 (PPT '인공신경망의 느낌?'을 선박 속도로 변경)")
print("=" * 60)

knot = 20.0
correct_kmh = 37.04

for w in [1.5, 1.7, 2.0, 1.85]:
    calculated = knot * w
    error = correct_kmh - calculated
    print(f"w = {w:<4} -> 계산값 = {calculated:6.2f}, 정답 = {correct_kmh}, 오차 = {error:7.3f}")

# 오차를 보고 w를 조금씩 고치는 과정(학습)을 자동화
print("\n[오차를 이용해 w를 자동으로 조정하는 학습]")
w = 1.5
lr = 0.00125  # 학습률(한 번에 얼마나 고칠지)
history_w, history_err = [], []
for step in range(1, 31):
    pred = knot * w
    err = correct_kmh - pred
    w = w + lr * err * knot        # 오차가 클수록 많이 수정
    history_w.append(w)
    history_err.append(err)
    if step in (1, 2, 3, 5, 10, 20, 30):
        print(f"step {step:2d}: w = {w:.5f}, 오차 = {err:8.4f}")
print(f"최종 w = {w:.5f}  (실제 환산계수 1.852)")

# ------------------------------------------------------------------
# 실습 2. 활성화 함수: 계단함수(step) vs 시그모이드(sigmoid)
#   PPT: sigma(x) = 1/(1+e^-x),  sigma'(x) = sigma(x)(1-sigma(x))
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("실습 2. 활성화 함수 (PPT '비선형함수의 필요')")
print("=" * 60)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def step_function(x, threshold=0.0):
    return np.where(x >= threshold, 1.0, 0.0)


for x in [-5, -1, 0, 1, 5]:
    print(f"x = {x:>2}: step = {int(step_function(np.array(x))):d}, "
          f"sigmoid = {sigmoid(x):.4f}, sigmoid' = {sigmoid_prime(x):.4f}")

xs = np.linspace(-8, 8, 400)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].plot(xs, step_function(xs), color="black")
axes[0].set_title("Step function (threshold=0)")
axes[1].plot(xs, sigmoid(xs), color="tab:blue")
axes[1].set_title("Sigmoid  1/(1+e^-x)")
axes[2].plot(xs, sigmoid_prime(xs), color="tab:red")
axes[2].set_title("Sigmoid derivative")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.set_xlabel("input")
axes[0].set_ylabel("output")
plt.tight_layout()
plt.savefig("activation_functions.png", dpi=130)
plt.close()
print("그래프 저장: activation_functions.png")

# ------------------------------------------------------------------
# 실습 3. 투입변수 3개 -> 은닉층 3개 -> 출력 1개 (PPT '개념도')
#   입력->은닉 가중치 w(9) = 3x3, 은닉->출력 가중치 w(3) = 3x1
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("실습 3. 3-3-1 신경망 순전파 (PPT '투입변수 3개에 따른 개념도')")
print("=" * 60)

x_in = np.array([1.0, 2.0, 3.0])            # PPT 그림의 입력 1, 2, 3
W1 = np.random.randn(3, 3) * 0.5            # w(9): 입력층 -> 은닉층
W2 = np.random.randn(3, 1) * 0.5            # w(3): 은닉층 -> 출력층

print(f"입력층 -> 은닉층 가중치 개수 w(9) = {W1.size}")
print(f"은닉층 -> 출력층 가중치 개수 w(3) = {W2.size}")

hidden_in = x_in @ W1                       # 가중합 (Sum)
hidden_out = sigmoid(hidden_in)             # 활성화 함수
y_hat = sigmoid(hidden_out @ W2)
print(f"은닉층 가중합 : {np.round(hidden_in, 4)}")
print(f"은닉층 출력 a1,a2,a3 : {np.round(hidden_out, 4)}")
print(f"최종 출력 y : {y_hat.item():.4f}")

# ------------------------------------------------------------------
# 실습 4. 몸무게 추정 (PPT '인공신경망 개념을 활용한 몸무게 추정')
#   입력: 키, 허리, 머리둘레 / 출력: 몸무게  (가상 데이터로 역전파 학습)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("실습 4. 신경망으로 몸무게 추정 (가상 데이터, 역전파 학습)")
print("=" * 60)

n = 300
height = np.random.uniform(150, 190, n)      # 키(cm)
waist = np.random.uniform(60, 100, n)        # 허리(cm)
head = np.random.uniform(52, 60, n)          # 머리둘레(cm)
# 비선형 관계를 가진 가상의 '정답' 몸무게
weight = (0.35 * (height - 100) + 0.0012 * waist ** 2 + 0.3 * head
          + np.random.normal(0, 1.0, n))

X = np.column_stack([height, waist, head])
Y = weight.reshape(-1, 1)

# 표준화 (입력/출력 스케일 맞추기)
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
Y_mean, Y_std = Y.mean(), Y.std()
Xn = (X - X_mean) / X_std
Yn = (Y - Y_mean) / Y_std

# 학습/검증 분리 (80:20)
idx = np.random.permutation(n)
tr, te = idx[:240], idx[240:]
Xtr, Ytr, Xte, Yte = Xn[tr], Yn[tr], Xn[te], Yn[te]

# 네트워크: 입력 3 -> 은닉 4(시그모이드) -> 출력 1(선형)   (PPT 그림과 동일 구조)
H = 4
W1 = np.random.randn(3, H) * 0.5
b1 = np.zeros((1, H))
W2 = np.random.randn(H, 1) * 0.5
b2 = np.zeros((1, 1))
lr = 0.1
epochs = 3000
loss_hist = []

for ep in range(epochs):
    # 순전파
    z1 = Xtr @ W1 + b1
    a1 = sigmoid(z1)
    pred = a1 @ W2 + b2
    err = pred - Ytr
    loss = np.mean(err ** 2)
    loss_hist.append(loss)

    # 역전파
    d_pred = 2 * err / len(Xtr)
    dW2 = a1.T @ d_pred
    db2 = d_pred.sum(axis=0, keepdims=True)
    d_a1 = d_pred @ W2.T
    d_z1 = d_a1 * a1 * (1 - a1)              # 시그모이드 미분 사용
    dW1 = Xtr.T @ d_z1
    db1 = d_z1.sum(axis=0, keepdims=True)

    # 가중치 갱신
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if ep in (0, 99, 499, 999, 1999, 2999):
        print(f"epoch {ep + 1:4d}: 학습 손실(MSE, 표준화 단위) = {loss:.5f}")


def predict(Xin):
    return sigmoid(Xin @ W1 + b1) @ W2 + b2


pred_te = predict(Xte) * Y_std + Y_mean
true_te = Yte * Y_std + Y_mean
rmse = np.sqrt(np.mean((pred_te - true_te) ** 2))
ss_res = np.sum((true_te - pred_te) ** 2)
ss_tot = np.sum((true_te - true_te.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"\n[검증 데이터 {len(te)}개] RMSE = {rmse:.3f} kg, R^2 = {r2:.3f}")

# 선형회귀와 비교 (PPT '전통적 선형회귀분석의 한계')
A = np.column_stack([Xtr, np.ones(len(Xtr))])
coef, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
lin_pred = (np.column_stack([Xte, np.ones(len(Xte))]) @ coef) * Y_std + Y_mean
lin_rmse = np.sqrt(np.mean((lin_pred - true_te) ** 2))
lin_r2 = 1 - np.sum((true_te - lin_pred) ** 2) / ss_tot
print(f"[선형회귀 비교] RMSE = {lin_rmse:.3f} kg, R^2 = {lin_r2:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(loss_hist, color="tab:blue")
axes[0].set_yscale("log")
axes[0].set_title("Training loss (MSE)")
axes[0].set_xlabel("epoch")
axes[0].grid(alpha=0.3)
axes[1].scatter(true_te, pred_te, s=18, color="tab:green")
lims = [true_te.min(), true_te.max()]
axes[1].plot(lims, lims, "k--", linewidth=1)
axes[1].set_title("Predicted vs actual weight (test)")
axes[1].set_xlabel("actual (kg)")
axes[1].set_ylabel("predicted (kg)")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("weight_estimation.png", dpi=130)
plt.close()
print("그래프 저장: weight_estimation.png")

# ------------------------------------------------------------------
# 실습 5. 흑백 이미지의 행렬 변환 (PPT '흑백 이미지의 행렬 데이터 변환')
#   숫자 '1' 모양을 14x14 행렬(0=흰색, 1=검정)로 만들고 1차원 입력으로 펼치기
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("실습 5. 흑백 이미지 -> 행렬 -> 입력 벡터")
print("=" * 60)

img = np.zeros((14, 14))
# (행, 열): 값  -- PPT 슬라이드의 14x14 행렬을 그대로 옮김
cells = {
    2: {6: 0.6, 7: 0.8},
    3: {6: 0.7, 7: 1.0},
    4: {6: 0.7, 7: 1.0},
    5: {6: 0.5, 7: 1.0, 8: 0.4},
    6: {7: 1.0, 8: 0.4},
    7: {7: 1.0, 8: 0.4},
    8: {7: 1.0, 8: 0.7},
    9: {7: 1.0, 8: 1.0},
    10: {7: 0.9, 8: 1.0, 9: 0.1},
    11: {7: 0.3, 8: 1.0, 9: 0.1},
}
for r, cols in cells.items():
    for c, v in cols.items():
        img[r, c] = v

print("행렬 크기:", img.shape)
for row in img:
    print(" ".join(f"{v:3.1f}" if v > 0 else "  0" for v in row))

flat = img.reshape(-1)
print(f"\n1차원으로 펼친 입력 벡터 길이 = {flat.size} (14x14)")
print("-> 이 196개 값이 입력층의 입력변수 196개가 됨")

plt.figure(figsize=(3.2, 3.2))
plt.imshow(img, cmap="gray_r")
plt.title("digit '1' as 14x14 matrix")
plt.axis("off")
plt.tight_layout()
plt.savefig("digit_matrix.png", dpi=130)
plt.close()
print("그래프 저장: digit_matrix.png")
