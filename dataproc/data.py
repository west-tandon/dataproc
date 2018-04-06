"""General data-related utilities."""
import functools
import operator
import pandas as pd
import numpy as np
import sklearn.preprocessing  # pylint: disable=import-error


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


def scale_features(data, *, scaler=sklearn.preprocessing.StandardScaler(),
                   exclude=set(['query', 'label'])):
    """Scale features according to `scaler` excluding `exclude` columns."""
    columns = data.columns
    features = list(set(columns).difference(exclude))
    if not features:
        return data
    non_features = list(set(columns).intersection(exclude))
    not_scaled = data.as_matrix(non_features) if non_features else None
    scaled = data.as_matrix(features)
    scaled = scaler.fit_transform(scaled)
    if not_scaled is not None:
        result = pd.DataFrame(np.concatenate([not_scaled, scaled], axis=1),
                              columns=non_features + features)
    else:
        result = pd.DataFrame(scaled, columns=features)
    return result[columns]


def to_trec(results, path, cutoff=1000):
    """Store `results` as a file in `trec_eval` format."""
    results['rank'] = (results.groupby('query')['score']
                       .rank(ascending=False, method='first')
                       .astype(np.int)) - 1
    results['iter'] = 'Q0'
    results['run_id'] = 'null'
    results = results[results['rank'] < cutoff].sort_values(['query', 'rank'])
    (results[['query', 'iter', 'title', 'rank', 'score', 'run_id']]
     .to_csv(path, header=False, sep='\t', index=False))


def to_svmrank(data, path, *, label='shard_score'):
    """Store `data` as a file in SVM-rank format."""
    svmdata = data.copy()
    if 'query' in svmdata.columns:
        svmdata.rename(columns={'query': 'qid'}, inplace=True)
    ensure_has_columns(svmdata, ['qid', label])
    feature_columns = list(set(svmdata.columns).difference(['qid', label]))
    svmdata = svmdata[[label, 'qid'] + feature_columns]
    columns = [label, 'qid'] + list(range(1, len(feature_columns) + 1))
    for column in columns[1:]:
        svmdata[column] = svmdata[column].apply(lambda v, c=column: f'{c}:{v}')
    svmdata.to_csv(path, header=False, sep=' ', index=False)
