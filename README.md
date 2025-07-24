# Momentum Optimization Visualization

This repository visualizes the difference between Classical Momentum and Nesterov's Accelerated Gradient optimization methods, based on the paper:

**"On the importance of initialization and momentum in deep learning"** by Sutskever et al.

## Visualization

The visualization shows:
1. Optimization paths on a quadratic objective with different curvatures
2. Convergence rates of different optimization methods

![Momentum Comparison](./1%201.png)

## Key Findings Illustrated

- Nesterov's momentum reduces oscillations in high-curvature directions compared to Classical Momentum
- Both momentum methods outperform standard gradient descent
- The effective momentum coefficient varies with curvature in Nesterov's method

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Serkalem-negusse1/momentum-optimization-visualization.git
cd momentum-optimization-visualization
