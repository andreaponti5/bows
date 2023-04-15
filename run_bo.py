import gpytorch
import time
import torch
import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import InputDataWarning

import utils

from components import test_functions
from components.bo import init_data, init_model, optimize_acquisition

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
res_filepath = f"results/bo_{test_function_name}_{nvar}.json"
res = {"config": {"test_function": test_function_name, "nvar": nvar, "bounds": bounds,
                  "ndesign": ndesign, "niter": niter}}
keys = ["X", "Y", "times"]
utils.save_res(res_filepath, res)

acq_bounds = torch.stack([torch.full((nvar,), bounds[0]),
                          torch.full((nvar,), bounds[1])]).double()
function = getattr(test_functions, test_function_name)
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

# Loop over multiple runs
for trial in range(ntrial):
    print(f"\n***TRIAL {trial}***\n")
    utils.set_seed(trial)
    start, iter_times = time.time(), []

    # Initialize the design set
    train_X, train_Y, _ = init_data(ndesign, nvar, bounds, function)
    # Initialize the surrogate model
    gp, mll = init_model(train_X, train_Y, kernel)

    # Loop over BO iterations
    for it in range(niter):
        iter_times.append(time.time() - start)

        # Fit the surrogate model
        fit_gpytorch_mll(mll)

        # Optimize the acquisition function and get the candidate
        new_X = optimize_acquisition(gp, acq_bounds)
        new_Y, _ = function(new_X)
        new_Y = new_Y.reshape(-1, 1)

        # Add the new observation to the train set
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])

        # Update the surrogate model
        gp, mll = init_model(train_X, train_Y, kernel, gp.state_dict())
        print(f"Iteration {it};\t Time = {iter_times[-1]: .2f};\t Best seen = {train_Y.min(): .8f};")

    iter_times.append(time.time() - start)

    # Save the results
    res_trial = utils.res_dict(keys,
                               train_X.tolist(),
                               train_Y.flatten().tolist(),
                               iter_times)
    res["trial_" + str(trial)] = res_trial
    utils.save_res(res_filepath, res)
