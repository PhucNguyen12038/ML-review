import numpy as np

def cost(x):
    return x**2 + 10*np.sin(x)

def grad(x):
    return 2*x + 10*np.cos(x)

def GD_without_Momentum(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return x,it

# def has_converged(theta_new, grad):
#     return np.linalg.norm(grad(theta_new))/len(theta_new) < 1e-3

def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new)) < 1e-3

def GD_momentum(theta_init, grad, eta, gamma):
    # Suppose we want to store history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        if has_converged(theta_new, grad):
            break
        theta.append(theta_new)
        v_old = v_new
    return theta,it

def NAG(theta_init, grad, eta, gamma):
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1] - gamma*v_old)
        theta_new = theta[-1] - v_new
        if has_converged(theta_new, grad):
            break
        theta.append(theta_new)
        v_old = v_new
    return theta,it

def GD_NAG(w_init, grad, eta, gamma):
    w = [w_init]
    v = [np.zeros_like(w_init)]
    for it in range(100):
        v_new = gamma*v[-1] + eta*grad(w[-1] - gamma*v[-1])
        w_new = w[-1] - v_new
        if has_converged(w_new,grad):
            break
        w.append(w_new)
        v.append(v_new)
    return (w, it)

x0 = 5
lr = 0.1
gamma = 0.9
x1,it1 = GD_without_Momentum(x0,lr)
x2,it2 = GD_momentum(x0,grad,lr,gamma)
x3,it3 = GD_NAG(x0,grad,lr,gamma)
print('GD without Momentum x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('GD with Momentum x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
print('GD with NAG x3 = %f, cost = %f, obtained after %d iterations'%(x3[-1], cost(x3[-1]), it3))
