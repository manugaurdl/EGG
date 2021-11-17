# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import (
    DefaultDict,
    Dict,
    List,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

K = TypeVar("K")
V = TypeVar("V")


def default_dict(pairs: Iterable[Tuple[K, V]]) -> DefaultDict[K, V]:
    mapping = defaultdict(list)
    for key, val in pairs:
        mapping[key].append(val)
    return mapping


def csv_to_dict(
    file_path: Path,
    key_col: int = 0,
    value_col: int = 1,
    discard_header: bool = True,
    one_to_n_mapping: bool = False,
) -> Union[Dict[str, str], DefaultDict[str, str]]:
    table = read_csv(file_path, discard_header)
    # map(itemgetter(key_col, value_col), table)
    pairs = ((line[key_col], line[value_col]) for line in table)
    if one_to_n_mapping:
        return default_dict(pairs)
    return dict(pairs)


def multicolumn_csv_to_dict(
    file_path: Path,
    key_cols: Sequence = (0,),
    value_cols: Optional[Sequence] = None,
    discard_header: bool = True,
    one_to_n_mapping: bool = False,
):
    table = read_csv(file_path, discard_header)
    if not value_cols:
        value_cols = tuple(i for i in range(1, len(table[0])))
    # (tuple(line[i] for i in key_cols) for line in table)
    key_columns = map(itemgetter(*key_cols), table)
    value_columns = map(itemgetter(*value_cols), table)
    pairs = zip(key_columns, value_columns)
    if one_to_n_mapping:
        return default_dict(pairs)
    return dict(pairs)


def read_csv(file_path: Path, discard_header: bool = True) -> List[List[str]]:
    with open(file_path) as text_file:
        text = text_file.read()

    lines = text.split("\n")
    _ = lines.pop() if lines[-1] == "" else None  # pop final empty line if present
    print(f"Loaded file {file_path}, with {len(lines)} lines")
    table = (x.split(",") for x in lines)
    if discard_header:
        next(table)
    return list(table)
