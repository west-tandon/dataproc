"""General data-related utilities."""
import functools
import operator
import pandas as pd


def cartesian(ranges, names=None):
    """Generates a data frame that is a cartesian product of ranges."""
    if names is None:
        names = range(len(ranges))
    if not ranges:
        return pd.DataFrame()
    if len(ranges) == 1:
        return pd.DataFrame({names[0]: ranges[0]})
    remaining_size = functools.reduce(
        operator.mul, [len(r) for r in ranges[1:]], 1)
    return pd.concat([
        pd.concat([pd.DataFrame({names[0]: [n] * remaining_size}),
                   cartesian(ranges[1:], names[1:])], axis=1)
        for n in ranges[0]
    ]).reset_index(drop=True)

def ensure_has_columns(dataframe, columns):
    """Raises an assertion error if `dataframe` misses any of `columns`."""
    for column in columns:
        assert column in dataframe.columns, \
                f'Column {column} missing in data frame'
