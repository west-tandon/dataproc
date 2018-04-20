"""Parquet tools."""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np


def merge_index(left, right, on, how=None):
    leftdata = pq.read_table(left, columns=on).to_pandas()
    rightdata = pq.read_table(right, columns=on).to_pandas()
    # TODO CHECK IF SORTED
    leftdata.index.name = '__left_row_idx__'
    rightdata.index.name = '__right_row_idx__'
    leftdata.reset_index(inplace=True)
    rightdata.reset_index(inplace=True)
    if how is None:
        merged_index = pd.merge(leftdata, rightdata, on=on)
    else:
        merged_index = pd.merge(leftdata, rightdata, on=on, how=how)
    if not isinstance(on, list):
        on = [on]
    for column in on:
        del merged_index[column]
    return merged_index.fillna(-1).astype(np.int64)


def merge(left, right, target, on, how=None):
    """Performs an on-disk merge."""
    merged_index = merge_index(left, right, on, how)
    left_pq = pq.ParquetFile(left)
    right_pq = pq.ParquetFile(right)
    return None


def sort(source, target, on):
    assert 0 == 1
