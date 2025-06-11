# Refactoring Plan

This project currently stores nearly all utilities in **FinancialMachineLearning.py**. To make the library easier to maintain we will divide the functions into focused modules.

## Proposed File Structure

- `data_preparation.py` – routines to load and clean raw market data.
- `bar_sampling.py` – tick, volume and dollar bar generation utilities.
- `labeling.py` – event extraction, barrier placement and label creation.
- `fractional_diff.py` – fractional differencing and related utilities.
- `cross_validation.py` – PurgedKFold and CV scoring helpers.
- `feature_importance.py` – MDI/MDA/SFI feature importance implementations.
- `hyperparameter.py` – grid search and random search helpers.
- `bet_sizing.py` – bet size calculation and position management utilities.
- `plotting.py` – functions for visualizing bars, weights and statistics.
- `multiprocessing_utils.py` – generic multiprocessing helpers (mpPandasObj, mpJobList etc.).

Each notebook will import from these modules to keep code modular and manageable.
