"""Provides tools for selective search."""

import math
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

import dataproc.data
from dataproc.data import ensure_has_columns


#def evaluate_step(shards, results, measures, step, verbose=False,
#                  with_buckets=False):
#    """Evaluates a single step."""
#    if verbose:
#        print(step, ' ', end='')
#    group_by = ['query', 'shard']
#    if with_buckets:
#        group_by.append('bucket')
#    top_shards = (shards.groupby(group_by)
#                  .apply(lambda g: g[:step])
#                  .reset_index(drop=True))
#    results_from_shards = (pd.merge(top_shards, results, on=group_by)
#                           .sort_values(['query', 'global_rank']))
#    evaluated = results_from_shards.groupby('query').agg(measures)
#    evaluated.columns = evaluated.columns.droplevel(0)
#    evaluated['step'] = step
#    return evaluated.reset_index()
#
#
#def evaluate(shards, results, measures, num_shards, verbose=False, sort=False,
#             num_buckets=None):
#    """
#    Evaluates each selection step.
#
#    Parameters
#    ----------
#    shards : pd.DataFrame
#        This data frame determines the order of selecting shards.
#        The following columns must be present:
#            - `query`: query ID
#            - `shard`: shard ID
#        Optional:
#            - `shard_score`: if sort=True, the shards within a query
#              will be sorted by these scores (descending); otherwise
#              it is assumed that they are already sorted
#    results : pd.DataFrame
#        This data frame contains the results of querying shards.
#        The following columns must be present:
#            - `query`: query ID
#            - `shard`: shard ID
#    measures : dict
#    num_shards : int
#    verbose : bool
#    sort : bool
#    num_buckets : None or int
#        If None, regular selective search is assumed.
#        Otherwise, it is the number of buckets in each shard.
#    """
#    if verbose:
#        print('Evaluating steps: ')
#    if sort:
#        group_by = ['query']
#        if num_buckets is not None:
#            group_by.append('bucket')
#        shards.groupby(group_by, sort=True).apply(
#            lambda g: g.sort_values('shard_score', ascending=False))
#    num_steps = num_shards if num_buckets is None else num_shards * num_buckets
#    return (pd.concat([evaluate_step(shards, results, measures, step, verbose,
#                                     num_buckets is not None)
#                       for step in range(1, num_steps + 1)])
#            .sort_values(['query', 'step']))


def load_shard_selection(queries, nshards, shard_scores_path):
    """Loads shard scores for the given queries."""
    data = dataproc.data.cartesian([queries, range(nshards)],
                                   names=['query', 'shard'])
    sel = pd.read_csv(shard_scores_path, names=['shard_score'])
    data = pd.concat([data, sel], axis=1)
    data['rank'] = (data.groupby('query')['shard_score']
                    .rank(method='first', ascending=False) - 1)
    return data

def load_bucket_selection(queries, nshards, nbuckets, shard_scores_path):
    """Loads bucket scores for the given queries."""
    data = dataproc.data.cartesian([queries, range(nshards), range(nbuckets)],
                                   names=['query', 'shard', 'bucket'])
    sel = pd.read_csv(shard_scores_path, names=['shard_score'])
    data = pd.concat([data, sel], axis=1)
    data['rank'] = (data.groupby('query')['shard_score']
                    .rank(method='first', ascending=False) - 1)
    return data


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

def select(selection, results, budget):
    """Selects results from the top shards within `budget` according to
    `selection`."""
    ensure_has_columns(selection, ['rank', 'query', 'shard'])
    ensure_has_columns(results, ['score', 'query', 'shard'])
    selection.sort_values(['query', 'rank'], inplace=True)
    selection['cumulative_cost'] = selection.groupby('query')['cost'].cumsum()
    budget_condition = selection['cumulative_cost'] <= budget
    within_budget = selection[budget_condition][['query', 'shard', 'cost']]
    avg_cost = within_budget.groupby('query')['cost'].sum().mean()
    return (avg_cost,
            (pd.merge(results, within_budget, on=['query', 'shard'])
             .sort_values(['query', 'score'], ascending=[True, False])
             .reset_index(drop=True)))

def decayed_buckets(num_buckets, num_shards, decay_factor):
    """TODO"""
    assert 0 < decay_factor <= 1
    sel = []
    for _ in range(num_shards):
        sel.append(math.ceil(num_buckets))
        num_buckets *= decay_factor
    return sel

def select_with_decay(selection, results, budget, decay_factor):
    """Selects results from the top `t` shards according to `selection`.

    The number of buckets taken from a shard will decay according to
    `decay_factor`."""
    if decay_factor == 1:
        return select(selection, results, budget)
    ensure_has_columns(results, ['bucket'])
    num_buckets = results['bucket'].max() + 1
    shard_selection = selection[selection['rank'] < budget]
    bucket_selection = decayed_buckets(num_buckets, budget, decay_factor)
    for rank, buckets in enumerate(bucket_selection):
        shard_selection.loc[shard_selection['rank'] == rank, 'buckets'] = buckets
    shard_results = pd.merge(results,
                             shard_selection[['query', 'shard', 'buckets']],
                             on=['query', 'shard'])
    return (shard_results[shard_results['bucket'] < shard_results['buckets']]
            .drop(columns='buckets')
            .sort_values(['query', 'score'], ascending=[True, False])
            .reset_index(drop=True))

def calc_prefix_bucket_costs(bucket_rank):
    """Calculates costs of selecting a bucket and its predecessors."""
    bucket_rank['prefix_cost'] = (bucket_rank
                                  .groupby(['query', 'shard'])['cost']
                                  .cumsum())
    return bucket_rank

def resolve_bucket_selection(bucket_rank, budget):
    """Resolves the bucket selection based on ranked buckets and budget."""

    def process_row(row, budget, total_cost, shard_costs, buckets_selected):
        """Process a single row."""
        shard = int(row['shard'])
        bucket = int(row['bucket'])
        if bucket < buckets_selected[shard]:
            return shard, bucket, None
        cost = row['prefix_cost'] - shard_costs[shard]
        if total_cost + cost <= budget:
            return shard, int(row['bucket']), cost
        return shard, bucket, None

    def buckets_for_query(query, bucket_rank, budget):
        """Resolves for a single query."""
        total_cost = 0
        shard_costs = [0] * num_shards
        buckets_selected = [0] * num_shards
        for idx, row in bucket_rank.sort_values('rank').iterrows():
            if total_cost == budget:
                break
            shard, bucket, cost = process_row(row, budget, total_cost,
                                              shard_costs, buckets_selected)
            #print('query', query, 'shard: ', shard, 'bucket: ', bucket, 'cost: ', cost)
            #print(buckets_selected)
            if cost is not None:
                shard_costs[shard] += cost
                buckets_selected[shard] = bucket + 1
                total_cost += cost
        return total_cost, pd.concat(
            [pd.DataFrame({'query': query, 'shard': shard,
                           'bucket': range(int(buckets))})
             for shard, buckets in enumerate(buckets_selected)])

    bucket_rank = calc_prefix_bucket_costs(bucket_rank)
    num_shards = bucket_rank['shard'].max() + 1
    resolved = []
    cost = 0
    num = 0
    for query, query_group in bucket_rank.groupby('query'):
        query_cost, selected = buckets_for_query(query, query_group, budget)
        resolved.append(selected)
        cost += query_cost
        num += 1
    return cost / num, pd.concat(resolved)

def select_buckets(selection, results, budget):
    ensure_has_columns(selection, ['rank', 'query', 'shard', 'bucket'])
    ensure_has_columns(results, ['score', 'query', 'shard', 'bucket'])
    avg_cost, bucket_selection = resolve_bucket_selection(selection, budget)
    return avg_cost, (pd.merge(results,
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
