"""Unit tests for dataproc.parquet."""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                os.pardir))
from dataproc.parquet import merge, merge_index


@pytest.fixture
def left_df():
    return pd.DataFrame({
        'index_col': [0, 0, 2, 3, 4, 6, 8, 10, 12, 14],
        'left_only': range(10)
    })


@pytest.fixture
def right_df():
    return pd.DataFrame({
        'index_col': [0, 0, 4, 3, 2, 5, 7, 9, 11, 13],
        'right_only': range(10)
    })


@pytest.fixture
def left_parquet(tmpdir, left_df):
    path = tmpdir / 'left.parquet'
    table = pa.Table.from_pandas(left_df)
    writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table, 3)
    return str(path)


@pytest.fixture
def right_parquet(tmpdir, right_df):
    path = tmpdir / 'right.parquet'
    table = pa.Table.from_pandas(right_df)
    writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table, 3)
    return str(path)


def test_merge_index(tmpdir, left_parquet, right_parquet):
    inner_expected = pd.DataFrame({
        '__left_row_idx__':  [0, 0, 1, 1, 2, 3, 4],
        '__right_row_idx__': [0, 1, 0, 1, 4, 3, 2]
    })
    how_empty = merge_index(left_parquet, right_parquet, on=['index_col'])
    assert how_empty.equals(inner_expected)
    how_inner = merge_index(left_parquet, right_parquet, on=['index_col'],
                            how='inner')
    assert how_inner.equals(inner_expected)

    left_expected = pd.DataFrame({
        '__left_row_idx__':  [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        '__right_row_idx__': [0, 1, 0, 1, 4, 3, 2] + [-1] * 5
    })
    how_left = merge_index(left_parquet, right_parquet, on=['index_col'],
                           how='left')
    assert how_left.equals(left_expected)

    right_expected = pd.DataFrame({
        '__left_row_idx__':  [0, 1, 0, 1, 2, 3, 4] + [-1] * 5,
        '__right_row_idx__': [0, 0, 1, 1, 4, 3, 2, 5, 6, 7, 8, 9]
    })
    how_right = merge_index(left_parquet, right_parquet, on=['index_col'],
                           how='right')
    assert how_right.equals(right_expected)


#def test_merge(tmpdir, left_parquet, right_parquet):
#    target_path = str(tmpdir / 'target.parquet')
#    merge(left_parquet, right_parquet, target_path, on=['index_col'])
#    target = pq.read_table(target_path).to_pandas()
