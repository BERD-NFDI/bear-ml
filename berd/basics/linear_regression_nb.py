# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# noqa: D100

# %% [markdown]
# # Linear Regression with PyTorch
#
# In this notebook we give a small introduction about a few functionalities and
# flexibility of pytorch as computation engine and as autograd framework.
# The basis for our experiments is a simple linear regression with synthetic data.

# %%
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.distributions import Uniform

# %% [markdown]
# ## The Dataset
#
# For our synthetic dataset, we first sample some input data $X$
# uniformly in a given range.
# The target variable $y$ is obtained by the linear equation:
#
# $$
# y = \beta X + t + \epsilon ,
# $$
#
# with the slope $\beta$ and the intercept $t$ as our parameters to estimate and
# $\epsilon$ as uncertainty over our true outcome drawn from a Gaussian distribution.

# %%
torch.manual_seed(42)

# Set generation options
x_range = (-2, 2)
num_samples = 25
slope = 1.85
intercept = -0.5
noise_variance = 0.5

# Draw some x values in the given range
uniform_dist = Uniform(*x_range)
x = uniform_dist.sample((num_samples, 1))

# Compute the target value y and add a bit of Gaussian noise.
y = x * slope + intercept
y += torch.randn(25, 1) * noise_variance

# %% [markdown]
# Let's inspect our freshly computed dataset.

# %%
plt.scatter(x[:, 0], y[:, 0], c='b')
plt.plot(x[:, 0], x[:, 0] * slope + intercept, c='g')
plt.show()
plt.close()


# %% [markdown]
# Given our $X$ and $y$ data, we can now investigate various ways
# to infer $\beta$ and $t$.
#
# ## Closed Form Solution
#
# The linear regression model offers a closed form solution to solve for our parameters:
#
# $$
# \hat \beta = (X^TX)^{-1}X^Ty.
# $$
#
# Pytorch has the same functionality as numpy for matrix operations and
# can be used in the exact same way.
# A major advantage is the possibility to perform calculations on the
# GPU without much overhead.

# %%
def estimate_beta(x: Tensor, y: Tensor) -> Tensor:
    """Get beta coefficient over closed form solution."""
    return torch.linalg.multi_dot([torch.inverse(torch.matmul(x.T, x)), x.T, y])


beta_hat = estimate_beta(x, y)
print('The estimated beta is {:.3f}.'.format(float(beta_hat[0])))

# %% [markdown]
# Well, something is not right yet! The closed-form solution does
# not have a dedicated intercept term.
# This can be helped by a little trick:
# From now on we add a 1 as an additional input feature to every $x \in X$,
# which can be seen as a replacement of the bias.
# This also allows to reuse the closed-form function without any adaptions.

# %%
x = torch.cat([x, torch.ones(25, 1)], dim=1)
print(x[:5])

beta_hat = estimate_beta(x, y)
print(
    'The estimated slope is {:.3f} and the intercept is {:.3f}.'.format(
        float(beta_hat[0]), float(beta_hat[1])
    )
)


# %% [markdown]
# We check how well the parameters approximate the linear function in a plot:

# %%
def plot_linear_regression(
    x: Tensor,
    y: Tensor,
    slope_true: float,
    intercept_true: float,
    slope_pred: float,
    intercept_pred: float,
) -> None:
    """Plot the result of a linear regression with known true parameters."""
    plt.scatter(x[:, 0], y[:, 0], c='b')
    plt.plot(x[:, 0], x[:, 0] * slope_true + intercept_true, c='g', label='Truth')
    plt.plot(x[:, 0], x[:, 0] * slope_pred + intercept_pred, c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.show()
    plt.close()


plot_linear_regression(
    x,
    y,
    slope_true=slope,
    intercept_true=intercept,
    slope_pred=float(beta_hat[0]),
    intercept_pred=float(beta_hat[1]),
)

# %% [markdown]
# That looks pretty close! Note that deviation in the output of the prediction
# is directly influenced by the amount of $\epsilon$.
