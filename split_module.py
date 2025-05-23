# coding: utf-8
"""Dataset splitting utilities.

This module implements the SplitDataModule class described in the
repository documentation. It supports several splitting modes and
validates inputs according to the accompanying specification files.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any


# Exception definitions -----------------------------------------------------
class DatasetTooSmallException(Exception):
    """Raised when the input dataset has fewer than two rows."""


class FractionOutOfRangeException(Exception):
    """Raised when fraction is not within (0, 1)."""


class InvalidRegexException(Exception):
    """Raised when a regex pattern is invalid or not applicable."""


class MissingParameterException(Exception):
    """Raised when a required parameter for the chosen mode is missing."""


class InvalidExpressionException(Exception):
    """Raised when the expression string is malformed."""


class InvalidDatasetFormatException(Exception):
    """Raised when the dataset does not satisfy schema requirements."""


class RecommenderSplitException(Exception):
    """Raised when the recommender split cannot be performed."""


class StratificationKeyException(Exception):
    """Raised when the stratification key is invalid."""


class GeneralSplitException(Exception):
    """Fallback exception for unexpected errors during splitting."""


# Helper functions ---------------------------------------------------------
def _parse_fraction(value: Any) -> float:
    """Parse fraction which may be float or percentage string."""
    if isinstance(value, str):
        if value.endswith('%'):
            try:
                num = float(value[:-1]) / 100.0
            except ValueError:
                raise FractionOutOfRangeException(
                    "Fraction string must be a number followed by %" )
        else:
            try:
                num = float(value)
            except ValueError:
                raise FractionOutOfRangeException(
                    "Fraction string must represent a float" )
    else:
        num = float(value)
    if not 0 < num < 1:
        raise FractionOutOfRangeException(
            "Fraction of rows for split must be greater than 0 and less than 1")
    return num


def _to_numeric(val: Any) -> Any:
    """Try to convert string values to float for expression evaluation."""
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return val
    return val


# Main class ----------------------------------------------------------------
@dataclass
class SplitDataModule:
    splitting_mode: str = "행 분할"
    fraction: Any = 0.5
    randomize: bool = True
    random_seed: int | None = None
    stratified: bool = False
    stratify_key: str | None = None
    regex: str | None = None
    target_column: str | None = None
    expression: str | None = None

    def split(self, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset according to current configuration."""
        if not dataset or len(dataset) < 2:
            raise DatasetTooSmallException(
                "Input dataset must contain at least two rows for splitting.")

        mode = self.splitting_mode
        try:
            if mode in ("행 분할", "rows"):
                return self._split_rows(dataset)
            elif mode in ("정규식 분할", "regex"):
                return self._split_by_regex(dataset)
            elif mode in ("조건식 분할", "expression"):
                return self._split_by_expression(dataset)
            elif mode in ("추천 분할", "recommender"):
                return self._split_recommender(dataset)
            else:
                raise ValueError(f"Unsupported splitting mode: {mode}")
        except Exception as exc:  # catch and rethrow as general if unknown
            if isinstance(exc, (DatasetTooSmallException,
                               FractionOutOfRangeException,
                               InvalidRegexException,
                               MissingParameterException,
                               InvalidExpressionException,
                               InvalidDatasetFormatException,
                               RecommenderSplitException,
                               StratificationKeyException,
                               ValueError)):
                raise
            raise GeneralSplitException(str(exc)) from exc

    # ------------------------------------------------------------------
    def _split_rows(self, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        frac = _parse_fraction(self.fraction)
        data = list(dataset)
        if self.randomize:
            rnd = random.Random(self.random_seed)
            rnd.shuffle(data)

        if self.stratified:
            if not self.stratify_key or self.stratify_key not in dataset[0]:
                raise StratificationKeyException(
                    "Stratification key is missing or invalid for stratified split.")
            groups: Dict[Any, List[Dict[str, Any]]] = {}
            for row in data:
                key = row.get(self.stratify_key)
                groups.setdefault(key, []).append(row)
            left: List[Dict[str, Any]] = []
            right: List[Dict[str, Any]] = []
            for rows in groups.values():
                count = int(len(rows) * frac)
                if self.randomize:
                    rnd.shuffle(rows)
                left.extend(rows[:count])
                right.extend(rows[count:])
            return left, right
        else:
            cut = int(len(data) * frac)
            left = data[:cut]
            right = data[cut:]
            return left, right

    # ------------------------------------------------------------------
    def _split_by_regex(self, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.regex or not self.target_column:
            raise MissingParameterException("regex and target_column are required for regex split mode")
        if self.target_column not in dataset[0]:
            raise InvalidRegexException("Target column not found in dataset")
        try:
            pattern = re.compile(self.regex)
        except re.error as exc:
            raise InvalidRegexException(str(exc)) from exc
        left: List[Dict[str, Any]] = []
        right: List[Dict[str, Any]] = []
        for row in dataset:
            val = str(row.get(self.target_column, ""))
            if pattern.search(val):
                left.append(row)
            else:
                right.append(row)
        return left, right

    # ------------------------------------------------------------------
    def _split_by_expression(self, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.expression:
            raise MissingParameterException("expression is required for expression split mode")
        left: List[Dict[str, Any]] = []
        right: List[Dict[str, Any]] = []
        for row in dataset:
            env = {k: _to_numeric(v) for k, v in row.items()}
            try:
                result = bool(eval(self.expression, {}, env))
            except Exception as exc:
                raise InvalidExpressionException(str(exc)) from exc
            if result:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # ------------------------------------------------------------------
    def _split_recommender(self, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if len(dataset[0]) != 3:
            raise InvalidDatasetFormatException(
                "Recommender split expects dataset with exactly 3 columns")
        frac = _parse_fraction(self.fraction)
        rnd = random.Random(self.random_seed)
        data = list(dataset)
        rnd.shuffle(data)
        cut = int(len(data) * frac)
        train = data[:cut]
        test = data[cut:]
        # ensure every user and item appears in train
        user_key, item_key, _ = list(dataset[0].keys())
        users_in_train = {row[user_key] for row in train}
        items_in_train = {row[item_key] for row in train}
        for row in test:
            if row[user_key] not in users_in_train or row[item_key] not in items_in_train:
                raise RecommenderSplitException(
                    "Each user and item must appear in training set at least once")
        return train, test


# Simple demonstration when run as a script ---------------------------------
if __name__ == "__main__":
    import csv
    import sys
    if len(sys.argv) < 2:
        print("Usage: python split_module.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    mod = SplitDataModule()
    left, right = mod.split(data)
    print(f"left rows: {len(left)}")
    print(f"right rows: {len(right)}")
