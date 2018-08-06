"""Provides tools for selective search."""

import math
import pandas as pd
import pyarrow.parquet as pq

import dataproc.data
from dataproc.data import ensure_has_columns


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


def select(selection, results, budget, nonempty=False):
    """Selects results from the top shards within `budget` according to
    `selection`."""
    ensure_has_columns(selection, ['rank', 'query', 'shard'])
    ensure_has_columns(results, ['score', 'query', 'shard'])
    selection.sort_values(['query', 'rank'], inplace=True)
    selection['cumulative_cost'] = selection.groupby('query')['cost'].cumsum()
    budget_condition = selection['cumulative_cost'] <= budget
    if nonempty:
        budget_condition |= selection['cumulative_cost'] == selection['cost']
    within_budget = selection[budget_condition]
    return (within_budget,
            pd.merge(results, within_budget[['query', 'shard', 'cost']],
                     on=['query', 'shard'])
            .sort_values(['query', 'score'], ascending=[True, False])
            .reset_index(drop=True))


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
        shard_selection.loc[
            shard_selection['rank'] == rank, 'buckets'] = buckets
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
        for _, row in bucket_rank.sort_values('rank').iterrows():
            if total_cost == budget:
                break
            shard, bucket, cost = process_row(row, budget, total_cost,
                                              shard_costs, buckets_selected)
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
    """Selects buckets."""
    ensure_has_columns(selection, ['rank', 'query', 'shard', 'bucket'])
    ensure_has_columns(results, ['score', 'query', 'shard', 'bucket'])
    _, bucket_selection = resolve_bucket_selection(selection, budget)
    bucket_selection = pd.merge(bucket_selection, selection,
                                on=['query', 'shard', 'bucket'])
    data = (pd.merge(results,
                     bucket_selection[['query', 'shard', 'bucket']],
                     on=['query', 'shard', 'bucket'])
            .sort_values(['query', 'score'], ascending=[True, False])
            .reset_index(drop=True))
    return bucket_selection, data


def apply_cost_model(data, cost_model, *, score_column='shard_score'):
    """Modifies `data` according to `cost_model`.

    If `cost_model` is None, no changes are made.
    Otherwise, it is joined with `data` and each shard score is divided
    by its cost. These can be either query-dependend or independent."""
    if cost_model is not None:
        data = pd.merge(data, cost_model).sort_values(['query', 'shard'])
        data.reset_index(inplace=True, drop=True)
        data.loc[data['cost'] > 0, score_column] /= data['cost']
        data.drop(columns=['cost'], inplace=True)
    return data
