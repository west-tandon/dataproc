"""Provides tools for selective search."""

import math
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import sys

import dataproc.data
from dataproc.data import ensure_has_columns


def evaluate_step(shards, results, measures, step, verbose=False,
                  with_buckets=False):
    """Evaluates a single step."""
    if verbose:
        print(step, ' ', end='')
    group_by = ['query', 'shard']
    if with_buckets:
        group_by.append('bucket')
    top_shards = (shards.groupby(group_by)
                  .apply(lambda g: g[:step])
                  .reset_index(drop=True))
    results_from_shards = (pd.merge(top_shards, results, on=group_by)
                           .sort_values(['query', 'global_rank']))
    evaluated = results_from_shards.groupby('query').agg(measures)
    evaluated.columns = evaluated.columns.droplevel(0)
    evaluated['step'] = step
    return evaluated.reset_index()


def evaluate(shards, results, measures, num_shards, verbose=False, sort=False,
             num_buckets=None):
    """
    Evaluates each selection step.

    Parameters
    ----------
    shards : pd.DataFrame
        This data frame determines the order of selecting shards.
        The following columns must be present:
            - `query`: query ID
            - `shard`: shard ID
        Optional:
            - `shard_score`: if sort=True, the shards within a query
              will be sorted by these scores (descending); otherwise
              it is assumed that they are already sorted
    results : pd.DataFrame
        This data frame contains the results of querying shards.
        The following columns must be present:
            - `query`: query ID
            - `shard`: shard ID
    measures : dict
    num_shards : int
    verbose : bool
    sort : bool
    num_buckets : None or int
        If None, regular selective search is assumed.
        Otherwise, it is the number of buckets in each shard.
    """
    if verbose:
        print('Evaluating steps: ')
    if sort:
        group_by = ['query']
        if num_buckets is not None:
            group_by.append('bucket')
        shards.groupby(group_by, sort=True).apply(
            lambda g: g.sort_values('shard_score', ascending=False))
    num_steps = num_shards if num_buckets is None else num_shards * num_buckets
    return (pd.concat([evaluate_step(shards, results, measures, step, verbose,
                                     num_buckets is not None)
                       for step in range(1, num_steps + 1)])
            .sort_values(['query', 'step']))


def load_shard_selection(queries, nshards, shard_scores_path):
    """Loads shard scores for the given queries."""
    df = dataproc.data.cartesian([queries, range(nshards)], names=['query', 'shard'])
    sel = pd.read_csv(shard_scores_path, names=['shard_score'])
    df = pd.concat([df, sel], axis=1)
    df['rank'] = (df.groupby('query')['shard_score']
                  .rank(method='first', ascending=False) - 1)
    return df

def load_bucket_selection(queries, nshards, nbuckets, shard_scores_path):
    """Loads bucket scores for the given queries."""
    df = dataproc.data.cartesian([queries, range(nshards), range(nbuckets)],
                                 names=['query', 'shard', 'bucket'])
    sel = pd.read_csv(shard_scores_path, names=['shard_score'])
    df = pd.concat([df, sel], axis=1)
    df['rank'] = (df.groupby('query')['shard_score']
                  .rank(method='first', ascending=False) - 1)
    return df


def load_shard_results(basename, nshards, nbuckets=1):
    """
    Loads results of running queries on shards.

    Parameters
    ----------
    shards : str
        A base path to results files. The filenames are resolved as follows:
        {basename}#{shard}.results-{buckets}
        These are parquet files with the following columns:
            - int32 query
            - int32 rank
            - int64 ldocid
            - int64 gdocid
            - double score
            - int32 shard
            - int32 bucket
    nshards : int
    nbuckets : int
    """
    return pd.concat([
        pq.read_table(f'{basename}#{shard}.results-{nbuckets}').to_pandas()
        for shard in range(nshards)
    ])

def select(selection, results, t):
    """Selects results from the top `t` shards according to `selection`."""
    ensure_has_columns(selection, ['rank', 'query', 'shard'])
    ensure_has_columns(results, ['score', 'query', 'shard'])
    return (pd.merge(results,
                     selection[selection['rank'] < t][['query', 'shard']],
                     on=['query', 'shard'])
            .sort_values(['query', 'score'], ascending=[True, False])
            .reset_index(drop=True))

def decayed_buckets(num_buckets, num_shards, decay_factor):
    assert 0 < decay_factor <= 1
    sel = []
    for shard in range(num_shards):
        sel.append(math.ceil(num_buckets))
        num_buckets *= decay_factor
    return sel

def select_with_decay(selection, results, t, decay_factor):
    """Selects results from the top `t` shards according to `selection`.

    The number of buckets taken from a shard will decay according to
    `decay_factor`."""
    if decay_factor == 1:
        return select(selection, results, t)
    ensure_has_columns(results, ['bucket'])
    num_buckets = results['bucket'].max() + 1
    shard_selection = selection[selection['rank'] < t]
    bucket_selection = decayed_buckets(num_buckets, t, decay_factor)
    for rank, buckets in enumerate(bucket_selection):
        shard_selection.loc[shard_selection['rank'] == rank, 'buckets'] = buckets
    shard_results = pd.merge(results,
                             shard_selection[['query', 'shard', 'buckets']],
                             on=['query', 'shard'])
    return (shard_results[shard_results['bucket'] < shard_results['buckets']]
            .drop(columns='buckets')
            .sort_values(['query', 'score'], ascending=[True, False])
            .reset_index(drop=True))

def resolve_bucket_selection(selection, threshold):
    num_shards = selection['shard'].max() + 1
    resolved = []
    for query, query_group in selection.groupby('query'):
        buckets_selected = 0
        query_selection = [0] * num_shards
        for idx, row in query_group.sort_values('rank').iterrows():
            if buckets_selected == threshold:
                break
            shard = int(row['shard'])
            cost = row['bucket'] + 1 - query_selection[shard]
            if cost < 1:
                continue
            if buckets_selected + cost <= threshold:
                query_selection[shard] += cost
                buckets_selected += cost
        resolved.append(pd.concat(
            [pd.DataFrame({'query': query,
                           'shard': shard,
                           'bucket': range(int(buckets))})
             for shard, buckets in enumerate(query_selection)]))
    return pd.concat(resolved)

def select_buckets(selection, results, t):
    ensure_has_columns(selection, ['rank', 'query', 'shard', 'bucket'])
    ensure_has_columns(results, ['score', 'query', 'shard', 'bucket'])
    bucket_selection = resolve_bucket_selection(selection, t)
    return (pd.merge(results,
                     bucket_selection[['query', 'shard', 'bucket']],
                     on=['query', 'shard', 'bucket'])
            .sort_values(['query', 'score'], ascending=[True, False])
            .reset_index(drop=True))

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
