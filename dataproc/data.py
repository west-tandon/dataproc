"""General data-related utilities."""
import functools
import operator
import subprocess
import re
import pandas as pd
import pyarrow.parquet as pq
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


def scale_features(data, *, scaler=None,
                   exclude=set(['query', 'label'])):
    """Scale features according to `scaler` excluding `exclude` columns."""
    columns = data.columns
    features = list(set(columns).difference(exclude))
    if not features:
        return data, scaler
    non_features = list(set(columns).intersection(exclude))
    not_scaled = data.as_matrix(non_features) if non_features else None
    scaled = data.as_matrix(features)
    if scaler is None:
        scaler = sklearn.preprocessing.StandardScaler()
        scaled = scaler.transform(scaled)
    else:
        scaled = scaler.fit_transform(scaled)
    if not_scaled is not None:
        result = pd.DataFrame(np.concatenate([not_scaled, scaled], axis=1),
                              columns=non_features + features)
    else:
        result = pd.DataFrame(scaled, columns=features)
    return result[columns], scaler


def to_trec(results, path, cutoff=1000, write_mode='w'):
    """Store `results` as a file in `trec_eval` format."""
    results['rank'] = (results.groupby('query')['score']
                       .rank(ascending=False, method='first')
                       .astype(np.int)) - 1
    results['iter'] = 'Q0'
    results['run_id'] = 'null'
    results = results[results['rank'] < cutoff].sort_values(['query', 'rank'])
    (results[['query', 'iter', 'title', 'rank', 'score', 'run_id']]
     .to_csv(path, header=False, sep='\t', index=False, mode=write_mode))


def to_svmrank(data, path, *, label='shard_score'):
    """Store `data` as a file in SVM-rank format."""
    svmdata = data.copy()
    if 'query' in svmdata.columns:
        svmdata.rename(columns={'query': 'qid'}, inplace=True)
    ensure_has_columns(svmdata, ['qid', label])
    feature_columns = [col for col in svmdata.columns
                       if col not in {'qid', label}]
    svmdata = svmdata[[label, 'qid'] + feature_columns]
    columns = [label, 'qid'] + list(range(1, len(feature_columns) + 1))
    for col, mapped in zip(svmdata.columns[2:], columns[2:]):
        svmdata[col] = svmdata[col].apply(lambda v, c=mapped: f'{c}:{v}')
    svmdata['qid'] = svmdata['qid'].apply(lambda v: f'qid:{int(v)}')
    svmdata.to_csv(path, header=False, sep=' ', index=False)


def load_data(path):
    """Load CSV or Parquet data file depending on extension."""
    if path.endswith('.csv'):
        data = pd.read_csv(path)
    else:
        data = pq.read_table(path).to_pandas()
    return data


def trec_eval(in_path, qrels_path, measures=['p10', 'map']):
    """Delegates evaluation to `trec_eval` cmd tool."""
    out = subprocess.run(
        f'trec_eval -m all_trec -q {qrels_path} {in_path}'.split(' '),
        stdout=subprocess.PIPE)
    values = {measure: None for measure in measures}
    exprs = {measure: re.compile(measure + r'\s+all\s+(?P<score>[0-9]+\.[0-9]+).*')
             for measure in measures}
    for line in out.stdout.decode('UTF-8').splitlines():
        for measure in measures:
            match = exprs[measure].match(line)
            if match:
                values[measure] = float(match.group('score'))
                break
    return values
