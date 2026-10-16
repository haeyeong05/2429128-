# 데이터처리개론 - 인공신경망 기초 실습

데이터처리개론 4. 인공신경망 기초 강의 내용을 NumPy로 직접 구현한 실습 코드입니다.

## 파일 구성
| 파일 | 내용 |
|---|---|
| `single_neuron.py` | 뉴런 1개의 계산 과정: Inputs, Weights, Sum, Bias, Activation, Output |
| `neural_net_basics.py` | 실습 1~5: knot→km/h 뉴런, 활성화 함수, 3-3-1 신경망, 몸무게 추정, 이미지 행렬 |
| `nonlinear_demo.py` | 실습 6: 선형회귀 vs 신경망 (비선형 데이터) |
| `*.png` | 실행 결과 그래프 |

## 실행 방법
```bash
pip install numpy matplotlib
python single_neuron.py
python neural_net_basics.py
python nonlinear_demo.py
```

실행하면 같은 폴더에 그래프 이미지(`activation_functions.png`, `weight_estimation.png`, `digit_matrix.png`, `nonlinear_comparison.png`)가 생성됩니다.
