# SplitDataModule implementation based on planning documents
import csv
import random
import re
from typing import List, Dict, Tuple, Any, Optional

class DatasetTooSmallException(Exception):
    """Raised when the input dataset is too small to split."""

class FractionOutOfRangeException(Exception):
    """Raised when the split fraction is not between 0 and 1."""

class InvalidRegexException(Exception):
    """Raised when a regex pattern is invalid or cannot be applied."""

class MissingParameterException(Exception):
    """Raised when a required parameter is missing."""

class InvalidExpressionException(Exception):
    """Raised when an expression for conditional split is invalid."""

class InvalidDatasetFormatException(Exception):
    """Raised when the dataset format does not meet requirements."""

class RecommenderSplitException(Exception):
    """Generic error during recommender split."""

class StratificationKeyException(Exception):
    """Raised when stratification key is missing or invalid."""

class GeneralSplitException(Exception):
    """Generic split error."""

Record = Dict[str, Any]
Dataset = List[Record]

class SplitDataModule:
    """Data splitting utility supporting multiple modes."""
    def __init__(
        self,
        splitting_mode: str = "rows",
        fraction: float = 0.5,
        randomize: bool = True,
        random_seed: Optional[int] = None,
        stratified: bool = False,
        stratify_key: Optional[str] = None,
        regex: Optional[str] = None,
        target_column: Optional[str] = None,
        expression: Optional[str] = None,
    ) -> None:
        self.splitting_mode = splitting_mode
        self.randomize = randomize
        self.random_seed = random_seed
        self.stratified = stratified
        self.stratify_key = stratify_key
        self.regex = regex
        self.target_column = target_column
        self.expression = expression
        if isinstance(fraction, str):
            if fraction.endswith('%'):
                fraction = float(fraction.strip('%')) / 100.0
            else:
                fraction = float(fraction)
        self.fraction = fraction
        self._validate_init()

    def _validate_init(self) -> None:
        modes = {"rows", "regex", "expression", "recommender"}
        if self.splitting_mode not in modes:
            raise ValueError(f"Unsupported splitting mode: {self.splitting_mode}")
        if not 0 < self.fraction < 1:
            raise FractionOutOfRangeException(
                "Fraction of rows for split must be greater than 0 and less than 1 (0% < fraction < 100%)."
            )
        if self.stratified and not self.stratify_key:
            raise StratificationKeyException(
                "Stratification key is missing or invalid for stratified split."
            )
        if self.splitting_mode == "regex" and (not self.regex or not self.target_column):
            raise MissingParameterException(
                "Required parameter 'regex' or 'target_column' is missing for Regular Expression split mode."
            )
        if self.splitting_mode == "expression" and not self.expression:
            raise MissingParameterException(
                "Required parameter 'expression' is missing for Expression split mode."
            )

    def split(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        if len(dataset) < 2:
            raise DatasetTooSmallException(
                "Input dataset must contain at least two rows for splitting."
            )
        if self.randomize and self.random_seed is not None:
            random.seed(self.random_seed)
        if self.splitting_mode == "rows":
            return self._split_rows(dataset)
        if self.splitting_mode == "regex":
            return self._split_by_regex(dataset)
        if self.splitting_mode == "expression":
            return self._split_by_expression(dataset)
        if self.splitting_mode == "recommender":
            return self._split_recommender(dataset)
        raise GeneralSplitException("Unsupported splitting mode")

    def _split_rows(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        data = list(dataset)
        if self.randomize:
            random.shuffle(data)
        if self.stratified:
            return self._stratified_split(data)
        cut = int(len(data) * self.fraction)
        return data[:cut], data[cut:]

    def _stratified_split(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        key = self.stratify_key
        if key not in dataset[0]:
            raise StratificationKeyException(
                "Stratification key is missing or invalid for stratified split."
            )
        groups: Dict[Any, List[Record]] = {}
        for row in dataset:
            groups.setdefault(row.get(key), []).append(row)
        first: Dataset = []
        second: Dataset = []
        for group_rows in groups.values():
            if self.randomize:
                random.shuffle(group_rows)
            cut = int(len(group_rows) * self.fraction)
            first.extend(group_rows[:cut])
            second.extend(group_rows[cut:])
        return first, second

    def _split_by_regex(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        try:
            pattern = re.compile(self.regex)
        except re.error as e:
            raise InvalidRegexException(
                "The regular expression pattern is invalid or cannot be applied to the target column."
            ) from e
        if self.target_column not in dataset[0]:
            raise InvalidRegexException(
                "The regular expression pattern is invalid or cannot be applied to the target column."
            )
        match_set: Dataset = []
        non_match_set: Dataset = []
        for row in dataset:
            value = str(row.get(self.target_column, ""))
            if pattern.search(value):
                match_set.append(row)
            else:
                non_match_set.append(row)
        return match_set, non_match_set

    def _parse_expression(self) -> Tuple[str, str, float]:
        expr = self.expression.strip()
        m = re.match(r"(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)", expr)
        if not m:
            raise InvalidExpressionException(
                "The provided relational expression is invalid or references an unavailable column."
            )
        field, op, value = m.groups()
        try:
            value_num = float(value)
        except ValueError:
            raise InvalidExpressionException(
                "The provided relational expression is invalid or references an unavailable column."
            )
        return field, op, value_num

    def _split_by_expression(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        field, op, value = self._parse_expression()
        if field not in dataset[0]:
            raise InvalidExpressionException(
                "The provided relational expression is invalid or references an unavailable column."
            )
        true_set: Dataset = []
        false_set: Dataset = []
        for row in dataset:
            try:
                cell_value = float(row[field])
            except (KeyError, ValueError):
                raise InvalidExpressionException(
                    "The provided relational expression is invalid or references an unavailable column."
                )
            result = False
            if op == '==':
                result = cell_value == value
            elif op == '!=':
                result = cell_value != value
            elif op == '>':
                result = cell_value > value
            elif op == '>=':
                result = cell_value >= value
            elif op == '<':
                result = cell_value < value
            elif op == '<=':
                result = cell_value <= value
            if result:
                true_set.append(row)
            else:
                false_set.append(row)
        return true_set, false_set

    def _split_recommender(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        if len(dataset[0].keys()) != 3:
            raise InvalidDatasetFormatException(
                "Input dataset schema is invalid for the selected mode (expected 3 columns for recommender split)."
            )
        data = list(dataset)
        if self.randomize:
            random.shuffle(data)
        cut = int(len(data) * self.fraction)
        return data[:cut], data[cut:]

def read_csv(path: str) -> Dataset:
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def main() -> None:
    dataset = read_csv('data.csv')
    module = SplitDataModule(splitting_mode='rows', fraction=0.6, random_seed=42)
    train, test = module.split(dataset)
    print('TRAIN:', train)
    print('TEST:', test)

if __name__ == '__main__':
    main()
