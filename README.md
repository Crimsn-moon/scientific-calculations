# 🔬 Scientific Python & Calculations

A comprehensive collection of scientific computing, numerical analysis, and mathematical calculations implemented in Python. This repository covers a wide range of algorithms and methods used in mathematics, physics, engineering, and data science.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This repository serves as both a learning resource and a practical toolkit for scientific computing in Python. It demonstrates the implementation of various numerical methods, mathematical algorithms, and scientific simulations using industry-standard libraries.

## ✨ Features

### Numerical Methods
- **Integration**: Trapezoidal rule, Simpson's rule, Gaussian quadrature, Monte Carlo integration
- **Differentiation**: Forward/backward/central differences, Richardson extrapolation
- **Root Finding**: Bisection, Newton-Raphson, secant method, fixed-point iteration
- **Optimization**: Gradient descent, Newton's method, conjugate gradient

### Linear Algebra & Matrix Operations
- Matrix decompositions (LU, QR, SVD, Eigenvalue)
- Solving linear systems and least squares problems
- Matrix factorization and transformations
- Sparse matrix operations

### Physics Simulations
- **Classical Mechanics**: Projectile motion, pendulum dynamics, N-body problems
- **Electromagnetism**: Electric and magnetic field calculations, wave propagation
- **Thermodynamics**: Heat transfer, statistical mechanics
- **Quantum Mechanics**: Wave functions, Schrödinger equation solvers

### Statistical Analysis
- Descriptive statistics and data summarization
- Probability distributions (normal, binomial, Poisson, etc.)
- Hypothesis testing and confidence intervals
- Regression analysis (linear, polynomial, non-linear)
- Monte Carlo simulations

### Signal Processing
- Fourier transforms (FFT, DFT)
- Filtering (low-pass, high-pass, band-pass)
- Convolution and correlation
- Time series analysis and forecasting
- Spectral analysis

## 🛠️ Tech Stack

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computing and array operations
- **SciPy**: Scientific and technical computing
- **Matplotlib**: Data visualization and plotting
- **SymPy**: Symbolic mathematics
- **Jupyter**: Interactive notebooks for demonstrations

## 📦 Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Clone the Repository
```bash
git clone https://github.com/Crimsn-moon/scientific-calculations.git
cd scientific-calculations
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install numpy scipy matplotlib sympy jupyter pandas
```

## 🚀 Usage

### Running Python Scripts
```bash
python scripts/numerical_integration.py
```

### Using Jupyter Notebooks
```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open any `.ipynb` file to explore interactive examples.

### Importing as a Module
```python
from numerical_methods import integrate, differentiate
from physics_simulations import ProjectileMotion

# Example: Numerical integration
result = integrate.simpson(f, a, b, n=1000)

# Example: Physics simulation
sim = ProjectileMotion(v0=50, angle=45)
sim.run()
sim.plot()
```

## 📂 Modules
```
scientific-calculations/
├── numerical_methods/
│   ├── integration.py
│   ├── differentiation.py
│   ├── root_finding.py
│   └── optimization.py
├── linear_algebra/
│   ├── decomposition.py
│   ├── solvers.py
│   └── transformations.py
├── physics_simulations/
│   ├── mechanics.py
│   ├── electromagnetism.py
│   └── quantum.py
├── statistics/
│   ├── distributions.py
│   ├── hypothesis_testing.py
│   └── regression.py
├── signal_processing/
│   ├── fourier.py
│   ├── filters.py
│   └── time_series.py
├── notebooks/
│   └── [Interactive Jupyter notebooks]
└── examples/
    └── [Example scripts and demonstrations]
```

## 📊 Examples

### Numerical Integration
```python
import numpy as np
from numerical_methods.integration import simpson

# Define function to integrate
def f(x):
    return np.sin(x) * np.exp(-x)

# Integrate from 0 to π
result = simpson(f, 0, np.pi, n=1000)
print(f"Integral value: {result:.6f}")
```

### Matrix Decomposition
```python
import numpy as np
from linear_algebra.decomposition import lu_decompose

# Create a matrix
A = np.array([[4, 3], [6, 3]])

# Perform LU decomposition
L, U = lu_decompose(A)
print("Lower triangular matrix:\n", L)
print("Upper triangular matrix:\n", U)
```

### Physics Simulation
```python
from physics_simulations.mechanics import simulate_pendulum
import matplotlib.pyplot as plt

# Simulate a pendulum
time, theta = simulate_pendulum(length=1.0, angle0=np.pi/4, duration=10)

plt.plot(time, theta)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Pendulum Motion')
plt.show()
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation and tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [SymPy Documentation](https://docs.sympy.org/)

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on this repository.

---

**Happy Computing! 🚀**
