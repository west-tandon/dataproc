"""Unit tests for dataproc.selectivesearch."""

import os
import sys

import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                os.pardir))
from dataproc.selectivesearch import (select,
                                      select_with_decay,
                                      decayed_buckets,
                                      select_buckets,
                                      resolve_bucket_selection,
                                      calc_prefix_bucket_costs)

@pytest.fixture
def results():
    return pd.DataFrame({
        'query': [0] * 6 + [1] * 6,
        'shard': [0, 0, 1, 1, 2, 2] * 2,
        'bucket': [0, 1, 0, 1, 0, 1] * 2,
        'score': [3, 2, 5, 2, 4, 1] + [2, 2, 4, 3, 5, 5]
    })

@pytest.fixture
def selection():
    return pd.DataFrame({
        'query': [0, 0, 0] + [1, 1, 1],
        'shard': list(range(3)) * 2,
        'rank': [0, 2, 1] + [2, 1, 0],
        'cost': [1] * 6
    })

@pytest.fixture
def bucket_selection():
    return pd.DataFrame({
        'query': [0] * 6 + [1] * 6,
        'shard': [0, 0, 1, 1, 2, 2] * 2,
        'bucket': [0, 1] * 6,
        'rank': [0, 3, 1, 4, 2, 5] + [2, 0, 3, 1, 4, 5],
        'cost': [1] * 12
    })

def test_select_all(results, selection):
    cost, selected = select(selection, results.drop(columns='bucket'), 3)
    expected = (results.drop(columns='bucket')
                .sort_values(['query', 'score'], ascending=[True, False])
                .reset_index(drop=True))
    assert selected[['query', 'score', 'shard']].equals(expected)
    assert cost == 3

def test_select_one(results, selection):
    cost, selected = select(selection, results.drop(columns='bucket'), 1)
    expected = pd.DataFrame({
        'query': [0, 0] + [1, 1],
        'shard': [0, 0] + [2, 2],
        'score': [3, 2] + [5, 5]
    })
    assert selected[['query', 'score', 'shard']].equals(expected)
    assert cost == 1

def test_decayed_buckets():
    assert decayed_buckets(10, 5, 1) == [10] * 5
    assert decayed_buckets(10, 5, 0.9) == [10, 9, 9, 8, 7]
    assert decayed_buckets(2, 3, 0.5) == [2, 1, 1]
    assert decayed_buckets(2, 3, 0.6) == [2, 2, 1]
    with pytest.raises(AssertionError):
        decayed_buckets(10, 5, 0)
    with pytest.raises(AssertionError):
        decayed_buckets(10, 5, 1.1)

def test_select_with_decay(results, selection):
    selected = select_with_decay(selection, results, 3, 0.5)
    expected = pd.DataFrame({
        'query': [0] * 4 + [1] * 4,
        'shard': [1, 2, 0, 0] + [2, 2, 1, 0],
        'bucket': [0, 0, 0, 1] + [0, 1, 0, 0],
        'score': [5, 4, 3, 2] + [5, 5, 4, 2]
    })
    assert selected.equals(expected), selected

def test_select_buckets(results, bucket_selection):
    cost, selected = select_buckets(bucket_selection, results, 3)
    expected = (pd.DataFrame({
        'query':  [0, 0, 0] + [1, 1, 1],
        'shard':  [0, 1, 2] + [0, 0, 1],
        'bucket': [0, 0, 0] + [0, 1, 0],
        'score':  [3, 5, 4] + [2, 2, 4]
    }).sort_values(['query', 'score'], ascending=[True, False])
      .reset_index(drop=True))
    assert selected.equals(expected), selected
    assert cost == 3

def test_resolve_bucket_selection(bucket_selection):
    cost, contiguous_selection = resolve_bucket_selection(bucket_selection, 3)
    expected = pd.DataFrame({
        'query':  [0, 0, 0] + [1, 1, 1],
        'shard':  [0, 1, 2] + [0, 0, 1],
        'bucket': [0, 0, 0] + [0, 1, 0]
    })
    contiguous_selection.reset_index(inplace=True, drop=True)
    contiguous_selection.sort_values(['query', 'shard'], inplace=True)
    assert contiguous_selection.equals(expected)
    assert cost == 3

def test_calc_prefix_bucket_costs(bucket_selection):
    bucket_costs = calc_prefix_bucket_costs(bucket_selection)
    expected =  pd.DataFrame({
                'query': [0] * 6 + [1] * 6,
                'shard': [0, 0, 1, 1, 2, 2] * 2,
                'bucket': [0, 1] * 6,
                'rank': [0, 3, 1, 4, 2, 5] + [2, 0, 3, 1, 4, 5],
                'cost': [1] * 12,
                'prefix_cost': [1, 2] * 6
            })
    print(bucket_costs)
    print(expected)
    assert (bucket_costs.reindex(sorted(bucket_costs.columns), axis=1)
            .equals(expected.reindex(sorted(expected.columns), axis=1)))
