# CODEXExample

This repository demonstrates a simple data splitting module implemented
according to the project specification documents (`01.프로젝트계획서` ~
`06.예외정의서`).

## split_data_module.py

`SplitDataModule` splits a dataset into two parts using different modes:

- `rows`: random or sequential splitting based on a ratio.
- `regex`: split using a regular expression on a target column.
- `expression`: split using a relational expression such as `Price > 100`.

The module validates parameters and raises custom exceptions defined in
`06.예외정의서.txt` when misuse is detected.

### Example

```python
from split_data_module import Dataset, SplitDataModule
import csv

with open('data.csv') as f:
    reader = csv.DictReader(f)
    ds = Dataset(list(reader))

module = SplitDataModule(ds, split_ratio=0.8, splitting_mode='rows', random_seed=42)
train, test, _ = module.run()
print(len(train.rows), len(test.rows))
```


