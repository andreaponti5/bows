import torch


def alpine01(x, **kwargs):
    """
    https://al-roomi.org/benchmarks/unconstrained/n-dimensions/162-alpine-function-no-1
    sum_i abs(x_i * sin(x_i) + 0.1 * x_i)
    Each component assumes value in [0, 10]

    :param x: Tensor n*d dimension
    :return: Tensor[float]
    """
    assert (x >= -10).all() and (x <= 10).all()
    components = torch.abs(x * x.sin() + 0.1 * x)
    return torch.sum(components, dim=1), components


def michalewicz(x, **kwargs):
    """
    http://www.sfu.ca/~ssurjano/michal.html
    Each component assumes value in [0, 1]

    :param x: Tensor n*d dimension
    :return: Tensor[float]
    """
    assert (x >= 0).all() and (x <= torch.pi).all()
    m = 10 if kwargs.get("m") is None else kwargs.get("m")
    i = torch.arange(1, x.shape[1] + 1)
    components = x.sin() * torch.pow(((i * torch.pow(x, 2)) / torch.pi).sin(), 2 * m)
    return -torch.sum(components, dim=1), components


def vincent(x, **kwargs):
    """
    Each component assumes value in [-1, 1].

    :param x:
    :return:
    """
    assert (x >= 0.25).all() and (x <= 10).all()
    components = torch.sin(10 * torch.log(x))
    return torch.sum(components, dim=1), components
