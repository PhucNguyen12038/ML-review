from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
n = 10

X = np.random.rand(n, 1)
y = 4 + 3 * X + .2*np.random.randn(n, 1) # noise added

# Building Xbar
one = np.ones((X.shape[0],1))
# Add 1 to beginning of X, axis 1 indicates Xbar has 2 cols, if 0, Xbar has 1 col
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ',w_lr.T)

# Display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Draw the fitting line
plt.plot(X.T, y.T, 'b.')     # data
plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()

def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2;

def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
    return g

def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))

def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(10):
        w_new = w[-1] - eta*grad(w[-1]) # w_new has same size with w
        # Chỗ này chỉ để đảm bảo là trung bình của trị tuyệt đối grad tại từng thành phần là nhỏ hơn 1e-3.
        # assert that mean of abs(grad) at each element < 1e-3
        # Nếu không chia cho len(w_new) thì với ma trận lớn, mọi thành phần đều nhỏ nhưng tổng của chúng có thể lớn. Vậy nên ta cần tính trung bình.
        # If not, with big matrix, each element is small but sum is big
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it)

w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, 1)
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))