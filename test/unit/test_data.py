"""Unit tests for dataproc.data."""

import os
import sys

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                os.pardir))
from dataproc.data import (scale_features)

@pytest.fixture
def features():
    return pd.DataFrame({
        'query': [0, 1, 2, 3, 4],
        'label': [4, 3, 2, 1, 0],
        'f1':    [3, 3, 3, 3, 3],
        'f2':    [2, 1, 2, 1, 2]
    })


class FakeScaler():
    """Simply negate values."""

    def transform(self, data):
        return np.negative(data)

    def fit_transform(self, data):
        return self.transform(data)


def test_scale_features_default_exclude(features):
    scaled = scale_features(features, scaler=FakeScaler())
    assert scaled.equals(pd.DataFrame({
        'query': [0, 1, 2, 3, 4],
        'label': [4, 3, 2, 1, 0],
        'f1':    -features['f1'],
        'f2':    -features['f2']
    }))


def test_scale_features_empty_exclude(features):
    scaled = scale_features(features, scaler=FakeScaler(),
                            exclude=[])
    assert scaled.equals(pd.DataFrame({
        'query': -features['query'],
        'label': -features['label'],
        'f1':    -features['f1'],
        'f2':    -features['f2']
    }))


def test_scale_features_full_exclude(features):
    scaled = scale_features(features, scaler=FakeScaler(),
                            exclude=features.columns)
    assert scaled.equals(features)
