import json
import numpy as np
import warnings

from scipy.stats import pearsonr, ConstantInputWarning


warnings.filterwarnings("ignore", category=ConstantInputWarning)

res_dir = "results/"
function = "vincent"

for d in [5, 10, 15, 20]:
    res_file = f"bows_{function.lower()}_{d}.json"
    res = json.load(open(res_dir + res_file, "r"))
    trial_key = ["trial_" + str(trial) for trial in range(10)]

    corrs = []
    for trial in trial_key:
        corrs.append([pearsonr(x, h)[0] for x, h in zip(res[trial]["X"], res[trial]["H"])])

    print(f"{np.nanmean(np.absolute(corrs))}")
