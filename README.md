# Bayesian Optimization in Wasserstein Space
This repository contains the code of the algorithm BOWS used in the following paper:  

Candelieri, A., Ponti, A. & Archetti, F. **Wasserstein enabled Bayesian optimization of composite functions.** _J Ambient Intell Human Comput_ (2023). [https://doi.org/10.3390/math11102342](https://doi.org/10.1007/s12652-023-04640-7)

## Python dependencies
Use the `requirements.txt` file as reference.  
You can automatically install all the dependencies using the following command. 
````bash
pip install -r requirements.txt
````

## How to use the code
There are two main entrypoints:
- `run_bo.py`: run the experiments using the standard BO algorithm.
- `run_bows.py`: run the experiments using the BOWS algorithm.

In both scripts, it is possible to modify the test function as well as the number of variables.

## How to cite us
If you use this repository, please cite the following paper:
> [Candelieri, A., Ponti, A. & Archetti, F. Wasserstein enabled Bayesian optimization of composite functions. J Ambient Intell Human Comput (2023). https://doi.org/10.3390/math11102342](https://doi.org/10.3390/math11102342)

```
@Article{Candelieri2023,
  AUTHOR = {Candelieri, Antonio and Ponti, Andrea and Archetti, Francesco},
  TITLE = {Wasserstein enabled Bayesian optimization of composite functions},
  JOURNAL = {Journal of Ambient Intelligence and Humanized Computing},
  YEAR = {2023},
  URL = {https://doi.org/10.1007/s12652-023-04640-7},
  ISSN = {1868-5145},
  DOI = {10.1007/s12652-023-04640-7}
}
```
