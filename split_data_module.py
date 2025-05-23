class Dataset:
    """Simple dataset wrapper for tabular data."""
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

class Transformer:
    """Placeholder transformer object."""
    pass

class ModuleWrapper:
    def run(self):
        raise NotImplementedError()

class DatasetTooSmallException(Exception):
    pass

class FractionOutOfRangeException(Exception):
    pass

class InvalidRegexException(Exception):
    pass

class MissingParameterException(Exception):
    pass

class InvalidExpressionException(Exception):
    pass

class InvalidDatasetFormatException(Exception):
    pass

class RecommenderSplitException(Exception):
    pass

class StratificationKeyException(Exception):
    pass

class GeneralSplitException(Exception):
    pass

import random
import re
from typing import Tuple, List, Any

class SplitDataModule(ModuleWrapper):
    """Dataset splitting module supporting multiple modes."""

    def __init__(self,
                 dataset: Dataset,
                 split_ratio: float = 0.7,
                 splitting_mode: str = "rows",
                 randomize: bool = True,
                 random_seed: int = None,
                 stratified: bool = False,
                 stratify_key: str = None,
                 target_column: str = None,
                 regex: str = None,
                 expression: str = None,
                 time_col: str = None):
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.splitting_mode = splitting_mode
        self.randomize = randomize
        self.random_seed = random_seed
        self.stratified = stratified
        self.stratify_key = stratify_key
        self.target_column = target_column
        self.regex = regex
        self.expression = expression
        self.time_col = time_col
        self._validate_init()

    def _validate_init(self):
        if not 0 < self.split_ratio < 1:
            raise FractionOutOfRangeException(
                "Fraction of rows for split must be between 0 and 1")
        if self.stratified and not self.stratify_key:
            raise StratificationKeyException(
                "Stratification key is missing or invalid for stratified split")
        if self.splitting_mode == "regex" and (not self.regex or not self.target_column):
            raise MissingParameterException("regex split requires regex and target_column")
        if self.splitting_mode == "expression" and not self.expression:
            raise MissingParameterException("expression split requires expression")

    def run(self) -> Tuple[Dataset, Dataset, Transformer]:
        if len(self.dataset) < 2:
            raise DatasetTooSmallException(
                "Input dataset must contain at least two rows for splitting")
        if self.random_seed is not None:
            random.seed(self.random_seed)
        if self.splitting_mode == "rows":
            d1, d2 = self._split_rows()
        elif self.splitting_mode == "regex":
            d1, d2 = self._split_by_regex()
        elif self.splitting_mode == "expression":
            d1, d2 = self._split_by_expression()
        else:
            raise GeneralSplitException(f"Unknown splitting mode {self.splitting_mode}")
        return Dataset(d1), Dataset(d2), Transformer()

    def _split_rows(self) -> Tuple[List[Any], List[Any]]:
        data = list(self.dataset.rows)
        if self.randomize:
            random.shuffle(data)
        split_point = int(len(data) * self.split_ratio)
        return data[:split_point], data[split_point:]

    def _split_by_regex(self) -> Tuple[List[Any], List[Any]]:
        pattern = None
        try:
            pattern = re.compile(self.regex)
        except re.error:
            raise InvalidRegexException("Invalid regular expression pattern")
        data1, data2 = [], []
        for row in self.dataset.rows:
            value = str(row.get(self.target_column, ""))
            if pattern.search(value):
                data1.append(row)
            else:
                data2.append(row)
        return data1, data2

    def _split_by_expression(self) -> Tuple[List[Any], List[Any]]:
        # Very limited expression evaluation supporting column name then operator then number
        try:
            column, op, value = self._parse_expression(self.expression)
        except ValueError as e:
            raise InvalidExpressionException(str(e))
        data1, data2 = [], []
        for row in self.dataset.rows:
            cell = row.get(column)
            if cell is None:
                raise InvalidExpressionException("Expression references unknown column")
            try:
                cell_val = float(cell)
            except Exception:
                raise InvalidExpressionException("Column for expression must be numeric")
            if self._evaluate(cell_val, op, float(value)):
                data1.append(row)
            else:
                data2.append(row)
        return data1, data2

    @staticmethod
    def _parse_expression(expr: str):
        tokens = None
        for oper in [">=", "<=", ">", "<", "==", "!="]:
            if oper in expr:
                tokens = expr.split(oper)
                if len(tokens) == 2:
                    return tokens[0].strip(), oper, tokens[1].strip()
        raise ValueError("Invalid expression format")

    @staticmethod
    def _evaluate(val: float, op: str, rhs: float) -> bool:
        if op == ">":
            return val > rhs
        if op == "<":
            return val < rhs
        if op == ">=":
            return val >= rhs
        if op == "<=":
            return val <= rhs
        if op == "==":
            return val == rhs
        if op == "!=":
            return val != rhs
        raise ValueError("Unsupported operator")
