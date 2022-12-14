import numpy as np

x_train = [[1, 4, 6], [2,3,1]]
w1 = [[1,2], [2,2], [4,4]]
w2 = [[3,1], [4,6]]
b1 = b2 =[0]
a1 = [[0], [0]]

x_train = np.array(x_train)
w1 = np.array(w1)
w2 = np.array(w2)
b1 = np.array(b1)
b2 = np.array(b2)
a1 = np.array(a1)

def sigmoid(z):        # 안전한 np.exp() 계산을 위해
    a = 1 / (1 + np.exp(-z))              # 시그모이드 계산
    return a

def forpass(x):
    z1 = np.dot(x, w1) + b1        # 첫 번째 층의 선형 식을 계산합니다
    a1 = sigmoid(z1)               # 활성화 함수를 적용합니다
    z2 = np.dot(a1,w2) + b2  # 두 번째 층의 선형 식을 계산합니다.
    return z2

z = forpass(x_train)
print(z)