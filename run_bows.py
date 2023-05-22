import gpytorch
import time
import torch
import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import InputDataWarning

import utils

from components import test_functions
from components.bo import init_data, init_model, optimize_acquisition
from components.mapper import regressor, predict


warnings.filterwarnings("ignore", category=InputDataWarning)

# Problem configuration
test_function_name = "alpine01"
nvar = 20
bounds = (-10, 10)

# Experiment configuration
ntrial = 10
niter = 20 * nvar
ndesign = nvar

# Initialize the results dictionary
res_filepath = f"results/bows_{test_function_name}_{nvar}.json"
res = {"config": {"test_function": test_function_name, "nvar": nvar, "bounds": bounds,
                  "ndesign": ndesign, "niter": niter}}
keys = ["X", "Y", "H", "H_hat", "times"]
utils.save_res(res_filepath, res)

function = getattr(test_functions, test_function_name)
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

# Loop over multiple runs
for trial in range(ntrial):
    print(f"\n***TRIAL {trial}***\n")
    utils.set_seed(trial)
    start, iter_times = time.time(), []

    # Initialize the design set
    train_X, train_Y, train_H = init_data(ndesign, nvar, bounds, function)
    H_hat = torch.Tensor()
    # Initialize the surrogate model
    gp, mll = init_model(train_H, train_Y, kernel)
    # Initialize MLP to map data back in the input space
    regr = regressor(train_H, train_X, nvar)

    # Loop over BO iterations
    for it in range(niter):
        iter_times.append(time.time() - start)

        # Fit the surrogate model
        fit_gpytorch_mll(mll)

        # Weakly defined bounds for the acquisition function
        acq_bounds = torch.stack([train_H.min(dim=0).values, train_H.max(dim=0).values])
        # Optimize the acquisition function and get the candidate
        new_H_hat = optimize_acquisition(gp, acq_bounds)
        H_hat = torch.cat([H_hat, new_H_hat])

        # Map the component (H) back in the input space (X)
        new_X = predict(regr, new_H_hat, bounds)
        new_Y, new_H = function(torch.Tensor(new_X))
        new_Y = new_Y.reshape(-1, 1).double()

        # Add the new observation to the train set
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])
        train_H = torch.cat([train_H, new_H])

        # Update the MLP regression model with the new observation
        regr = regressor(train_H, train_X, nvar)
        # Update the surrogate model
        gp, mll = init_model(train_H, train_Y, kernel, gp.state_dict())

        print(f"Iteration {it};\t Time = {iter_times[-1]: .2f};\t Best seen = {train_Y.min(): .2f};")

    iter_times.append(time.time() - start)

    # Save the results
    res_trial = utils.res_dict(keys,
                               train_X.tolist(),
                               train_Y.flatten().tolist(),
                               train_H.tolist(), H_hat.tolist(),
                               iter_times)
    res["trial_" + str(trial)] = res_trial
    utils.save_res(res_filepath, res)
