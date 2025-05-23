# CODEXExample
This repository demonstrates several dataset splitting utilities.

The main module `split_module.py` defines the `SplitDataModule` class and a
set of helper functions.  They implement the behaviour described in the
project documentation files (01~06).

### Usage

```
python split_module.py data.csv
```

Additional helper functions can be imported and used programmatically:

- `split_dataset` – split into train/validation/test according to fractions.
- `k_fold_split` – generate K-fold cross validation sets.
- `sequential_split` – split data chronologically.
- `group_split` – split while keeping all rows of the same group together.


