import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound

from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood


def init_data(n_point, dimension, bounds, function, **kwargs):
    """

    :param n_point: number of initial point (design set)
    :param dimension: dimension of the input space (X)
    :param bounds: lower and upper bounds of the input space (X)
    :param function: function to evaluate
    :return:
    """
    train_X = torch.Tensor(np.array([
        (bounds[0] - bounds[1]) * np.random.rand(dimension) + bounds[1] for _ in range(n_point)
    ]))
    train_X = train_X.double()
    train_Y, train_H = function(train_X, **kwargs)
    train_Y = train_Y.reshape(-1, 1)
    train_Y = train_Y.double()
    return train_X, train_Y, train_H


def init_model(train_X, train_Y, kernel, state_dict=None):
    gp = SingleTaskGP(train_X, train_Y, covar_module=kernel)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    if state_dict is not None:
        gp.load_state_dict(state_dict)
    return gp, mll


def optimize_acquisition(gp, bounds):
    # Initialize and optimize the acquisition function
    UCB = UpperConfidenceBound(gp, beta=3, maximize=False)
    candidate, _ = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=50
    )
    return candidate.detach()
