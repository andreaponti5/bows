import torch

from sklearn.neural_network import MLPRegressor


def regressor(x, y, hidden_layes_size):
    return MLPRegressor(hidden_layer_sizes=[hidden_layes_size] * 5,
                        max_iter=10000,
                        random_state=1).fit(x, y)


def predict(model, x, bounds):
    y = model.predict(x)
    y[y < bounds[0]] = bounds[0]
    y[y > bounds[1]] = bounds[1]
    return torch.Tensor(y)
