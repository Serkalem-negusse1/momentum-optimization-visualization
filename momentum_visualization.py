"""
Visualization of Classical Momentum vs Nesterov's Accelerated Gradient
Based on the paper: "On the importance of initialization and momentum in deep learning"
by Sutskever et al.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def quadratic_objective(x, A, b):
    """Quadratic objective function: 0.5*x^T A x + b^T x"""
    return 0.5 * x.T @ A @ x + b.T @ x

def quadratic_gradient(x, A, b):
    """Gradient of quadratic objective"""
    return A @ x + b

def classical_momentum(x0, v0, A, b, learning_rate, momentum, n_iter):
    """Classical Momentum optimization"""
    x = x0.copy()
    v = v0.copy()
    trajectory = [x.copy()]
    
    for _ in range(n_iter):
        grad = quadratic_gradient(x, A, b)
        v = momentum * v - learning_rate * grad
        x = x + v
        trajectory.append(x.copy())
    
    return np.array(trajectory)

def nesterov_momentum(x0, v0, A, b, learning_rate, momentum, n_iter):
    """Nesterov's Accelerated Gradient optimization"""
    x = x0.copy()
    v = v0.copy()
    trajectory = [x.copy()]
    
    for _ in range(n_iter):
        # Lookahead position
        x_lookahead = x + momentum * v
        grad = quadratic_gradient(x_lookahead, A, b)
        v = momentum * v - learning_rate * grad
        x = x + v
        trajectory.append(x.copy())
    
    return np.array(trajectory)

def gradient_descent(x0, A, b, learning_rate, n_iter):
    """Standard gradient descent"""
    x = x0.copy()
    trajectory = [x.copy()]
    
    for _ in range(n_iter):
        grad = quadratic_gradient(x, A, b)
        x = x - learning_rate * grad
        trajectory.append(x.copy())
    
    return np.array(trajectory)

def main():
    # Set up the quadratic objective with different curvatures
    A = np.array([[100, 0],   # High curvature in first dimension
                  [0, 1]])    # Low curvature in second dimension
    b = np.array([0, 0])      # Zero linear term (minimum at origin)

    # Initial conditions
    x0 = np.array([-1.5, -1.5])
    v0 = np.array([0, 0])

    # Optimization parameters
    learning_rate = 0.02
    momentum = 0.9
    n_iter = 100

    # Run optimizers
    cm_traj = classical_momentum(x0, v0, A, b, learning_rate, momentum, n_iter)
    nag_traj = nesterov_momentum(x0, v0, A, b, learning_rate, momentum, n_iter)
    gd_traj = gradient_descent(x0, A, b, learning_rate, n_iter)

    # Create contour plot of the quadratic objective
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i,j], Y[i,j]])
            Z[i,j] = quadratic_objective(point, A, b)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Contour plot
    plt.subplot(1, 2, 1)
    levels = np.linspace(0, 200, 20)
    plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.plot(cm_traj[:,0], cm_traj[:,1], 'b-', label='Classical Momentum', linewidth=2)
    plt.plot(nag_traj[:,0], nag_traj[:,1], 'r-', label="Nesterov's", linewidth=2)
    plt.plot(gd_traj[:,0], gd_traj[:,1], 'g-', label='Gradient Descent', linewidth=2)
    plt.scatter([0], [0], c='k', marker='*', s=200, label='Optimum')
    plt.xlabel('x (high curvature direction)')
    plt.ylabel('y (low curvature direction)')
    plt.title('Optimization Paths')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Error vs iteration plot
    plt.subplot(1, 2, 2)
    cm_errors = [quadratic_objective(p, A, b) for p in cm_traj]
    nag_errors = [quadratic_objective(p, A, b) for p in nag_traj]
    gd_errors = [quadratic_objective(p, A, b) for p in gd_traj]

    plt.semilogy(cm_errors, 'b-', label='Classical Momentum', linewidth=2)
    plt.semilogy(nag_errors, 'r-', label="Nesterov's", linewidth=2)
    plt.semilogy(gd_errors, 'g-', label='Gradient Descent', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value (log scale)')
    plt.title('Convergence Rates')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('momentum_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()