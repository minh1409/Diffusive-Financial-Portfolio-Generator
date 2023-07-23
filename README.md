# Diffusive Financial Portfolio Generator

The project is on creating advanced control systems for portfolio optimization, utilizing a unique learning method for the derivative of the objective function. Initially created as an academic project in Jan/2022. 

## Introduction
The primary aim of our project is to empower our algorithm to generate a diverse set of potential portfolios, thereby facilitating optimal diversification. This is achieved by employing a novel method of learning the derivative of the objective function with respect to portfolios, which is diffusion modelling. This approach alleviates the non-differentiability of some financial functions, such as amount of profit. It also fixs an inherent flaw of financial deep reinforcement learning on generate greedy and risky portfolio.

Our project leverages data from [cophieu68.com](http://cophieu68.com), designed for integration with Amibroker software.

## Dependencies

This project is built upon several powerful Python libraries that enable high performance numerical computations, machine learning, and data visualizations. Here is a list of the key dependencies:

- **JAX**: Diffusion costs ton of computational resources. Jax is used for efficient automatic differentiation and optimization.
- **NumPy**: Numpy powers efficient numerical computations. It also used to store JAX weight files.
- **Matplotlib**: Empowers us to create static, animated, and interactive visualizations in Python.
- **TensorFlow**: An end-to-end open source platform used for machine learning. Mostly if requiring deep learning signal processing techniques.
- **Scikit-learn (sklearn)**: A robust tool for data mining and data analysis. We used it for extract linear features from signal.

You can install these dependencies using pip, a package installer for Python. Simply copy and paste the following command into your terminal:

```bash
pip install jax numpy matplotlib tensorflow sklearn
```

## Project Composition
- Run Data Generator.ipynb file first to pre-compute linear features for price signals.
- Then, run Model.ipynb file to define and train the Diffusion Model. The weight file is saved as numpy files (.npy)
- Finally, run Validate.ipynb for backtesting or computing portfolios for current date.
