"""Evaluate cross-validation folds for L2RR."""
import os
import re
import sys
import subprocess
import argparse
from collections import namedtuple
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import attr  # pylint: disable=import-error

from dataproc.data import to_trec, load_data
from dataproc.selectivesearch import (load_shard_selection,
                                      load_bucket_selection,
                                      select,
                                      select_buckets)


CostData = namedtuple(
    'CostData', ['method', 'size_costs', 'posting_costs',
                 'posting_costs_frac'])


def parse_cmd_args():
    """Returns parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate cross-validation folds for L2RR.')
    parser.add_argument('results', help='A parquet file containing results.')
    parser.add_argument('qrels', help='A TREC query-relevance file.')
    parser.add_argument('dir', help='Directory containing folds.')
    parser.add_argument('--folds', '-k', default=10, type=int,
                        help='The number of folds to run.')
    parser.add_argument('--budget', type=float, required=True,
                        help='The number of shards to select.')
    parser.add_argument('--header', '-H', action='store_true',
                        help='Print CSV header.')
    parser.add_argument('--buckets', default=1, type=int,
                        help='The number of buckets in experiment.')
    parser.add_argument('--shards', type=int, required=True,
                        help='The number of shards in experiment.')
    parser.add_argument('--selection-cost', required=True, dest='selcost',
                        action='store', choices=['N', 'S', 'P', 'X'],
                        help=('Type of selection cost:'
                              ' N -- each shard cost 1;'
                              ' S -- cost relative to shard size;'
                              ' P -- cost relative to posting count',
                              ' X -- cost relative to posting count fraction'))
    parser.add_argument('--size-costs', required=True,
                        help='A path to file containing size costs.')
    parser.add_argument('--posting-costs', required=True,
                        help='A path to file containing posting costs.')
    parser.add_argument('--posting-costs-frac', required=True,
                        help='A path to file containing posting costs (frac).')
    return parser.parse_args()


def trec_eval(*, fold_path, qrels_path, trec_in_path):
    """Delegates evaluation to `trec_eval` cmd tool."""
    out = subprocess.run(
        f'trec_eval -q {qrels_path} {trec_in_path}'.split(' '),
        stdout=subprocess.PIPE)
    trec_out_path = os.path.join(fold_path, 'trec.out')
    with open(trec_out_path, 'bw') as out_file:
        out_file.write(out.stdout)
    p10, map_measure = None, None
    for line in out.stdout.decode('UTF-8').splitlines():
        p10_re = re.compile(r'P_10\s+all\s+(?P<score>[0-9]+\.[0-9]+).*')
        map_re = re.compile(r'map\s+all\s+(?P<score>[0-9]+\.[0-9]+).*')
        p10_match = p10_re.match(line)
        map_match = map_re.match(line)
        if p10_match:
            p10 = float(p10_match.group('score'))
        if map_match:
            map_measure = float(map_match.group('score'))
    return p10, map_measure


def eval_fold(*, selection, results, qrels_path, fold_path, budget,
              nbuckets):
    """Evaluate a single cross-validation fold."""
    if nbuckets == 1:
        filtered_selection, selected = select(selection, results, budget,
                                              nonempty=False)
    else:
        filtered_selection, selected = select_buckets(selection, results,
                                                      budget)

    trec_in_path = os.path.join(fold_path, 'trec.in')
    to_trec(selected, trec_in_path)
    measures = trec_eval(fold_path=fold_path,
                         qrels_path=qrels_path,
                         trec_in_path=trec_in_path)
    return measures, filtered_selection


def load_selection(fold_path, nshards, nbuckets, cost_data):
    """Load shard/bucket selection data and merge with costs."""
    pred_path = os.path.join(fold_path, 'predictions')
    with open(os.path.join(fold_path, 'queries'), 'r') as query_file:
        queries = np.array([int(query) for query in query_file.read().split()])
    if nbuckets == 1:
        selection = load_shard_selection(queries, nshards, pred_path)
    else:
        selection = load_bucket_selection(
            queries, nshards, nbuckets, pred_path)

    unit_costs = pd.DataFrame({'shard': range(nshards), 'unit_cost': 1})
    selection = (pd.merge(selection, unit_costs)
                 .rename(columns={'cost': 'uniform_cost'}))
    selection = (pd.merge(selection, cost_data.size_costs, on='shard')
                 .rename(columns={'cost': 'size_cost'}))
    selection = (pd.merge(selection, cost_data.posting_costs)
                 .rename(columns={'cost': 'posting_cost'}))
    selection = (pd.merge(selection, cost_data.posting_costs_frac)
                 .rename(columns={'cost': 'posting_cost_frac'}))

    if cost_data.method == 'N':
        selection['cost'] = selection['unit_cost']
    elif cost_data.method == 'S':
        selection['cost'] = selection['size_cost']
    elif cost_data.method == 'P':
        selection['cost'] = selection['posting_cost']
    else:
        selection['cost'] = selection['posting_cost_frac']

    return selection


@attr.s
class Stats(object):
    """Statistics for a fold."""
    p10 = attr.ib(default=0.0)
    map1k = attr.ib(default=0.0)
    shards = attr.ib(default=0)
    size_cost = attr.ib(default=0.0)
    posting_cost = attr.ib(default=0.0)
    posting_cost_frac = attr.ib(default=0.0)

    def __add__(self, rhs):
        if not isinstance(rhs, Stats):
            raise AttributeError(f'Cannot add {rhs} to a Stats object')
        return Stats(
            self.p10 + rhs.p10,
            self.map1k + rhs.map1k,
            self.shards + rhs.shards,
            self.size_cost + rhs.size_cost,
            self.posting_cost + rhs.posting_cost,
            self.posting_cost_frac + rhs.posting_cost_frac)


def main():
    """Main function"""
    args = parse_cmd_args()
    results = pq.read_table(args.results).to_pandas()

    selections = []

    agg = Stats()
    if args.header:
        print('budget,fold,p10,map,shards,sizecost,postingcost,pcfrac')
    for fold in range(args.folds):
        fold_path = os.path.join(args.dir, f'fold-{fold}')
        selection = load_selection(
            fold_path, args.shards, args.buckets,
            CostData(args.selcost,
                     load_data(args.size_costs),
                     load_data(args.posting_costs),
                     load_data(args.posting_costs_frac))
        )
        (p10, map1k), filtered_selection = eval_fold(
            selection=selection,
            results=results,
            qrels_path=args.qrels,
            fold_path=fold_path,
            budget=args.budget,
            nbuckets=args.buckets)
        selections.append(filtered_selection)
        grouped = filtered_selection.groupby('query')
        fold_stats = Stats(
            p10,
            map1k,
            grouped['unit_cost'].sum().mean(),
            grouped['size_cost'].sum().mean(),
            grouped['posting_cost'].sum().mean(),
            grouped['posting_cost_frac'].sum().mean())
        agg += fold_stats
        print(args.budget, fold, fold_stats.p10, fold_stats.map1k,
              fold_stats.shards, fold_stats.size_cost, fold_stats.posting_cost,
              fold_stats.posting_cost_frac, sep=',')
    print('Avg. P@10: ', agg.p10 / args.folds, file=sys.stderr)
    print('Avg. MAP: ', agg.map1k / args.folds, file=sys.stderr)
    print('Avg. #shards: ', agg.shards / args.folds, file=sys.stderr)
    print('Avg. size cost: ', agg.size_cost / args.folds, file=sys.stderr)
    print('Avg. posting cost: ', agg.posting_cost / args.folds,
          file=sys.stderr)
    print('Avg. posting cost%: ', agg.posting_cost_frac / args.folds,
          file=sys.stderr)

    pd.concat(selections).to_csv(
        os.path.join(args.dir, f'selections{args.selcost}{args.budget}.csv'),
        index=False)


main()
