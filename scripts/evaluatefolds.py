"""Evaluate cross-validation folds for L2RR."""
import os
import re
import sys
import subprocess
import argparse
import numpy as np
import pyarrow.parquet as pq
import pandas as pd

from dataproc.data import to_trec
from dataproc.selectivesearch import (load_shard_selection,
                                      load_bucket_selection,
                                      select,
                                      select_buckets)


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
    parser.add_argument('--decay-factor', '-d', default=1, type=float,
                        help='The decay factor in selecting buckets.')
    parser.add_argument('--header', '-H', action='store_true',
                        help='Print CSV header.')
    parser.add_argument('--buckets', default=1, type=int,
                        help='The number of buckets in experiment.')
    parser.add_argument('--shards', type=int, required=True,
                        help='The number of shards in experiment.')
    parser.add_argument('--sizes',
                        help='A path to a CSV file containing sizes of all '
                             'shards to use for selecting (in absence of this '
                             'file, all shards are considered to be equal)')
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


def eval_fold(*, results, qrels_path, fold_path, budget, nshards, nbuckets,
              shard_costs):
    """Evaluate a single cross-validation fold."""
    pred_path = os.path.join(fold_path, 'predictions')
    with open(os.path.join(fold_path, 'queries'), 'r') as query_file:
        queries = np.array([int(query) for query in query_file.read().split()])

    if nbuckets == 1:
        selection = load_shard_selection(queries, nshards, pred_path)
    else:
        selection = load_bucket_selection(
            queries, nshards, nbuckets, pred_path)

    if shard_costs is None:
        shard_costs = pd.DataFrame({'shard': range(nshards), 'cost': 1})
    selection = pd.merge(selection, shard_costs, on='shard')

    if nbuckets == 1:
        avg_cost, selected = select(selection, results, budget)
    else:
        avg_cost, selected = select_buckets(selection, results,
                                            budget * nbuckets)

    trec_in_path = os.path.join(fold_path, 'trec.in')
    to_trec(selected, trec_in_path)
    measures = trec_eval(fold_path=fold_path,
                         qrels_path=qrels_path,
                         trec_in_path=trec_in_path)
    return measures, avg_cost / nbuckets


def main():
    """Main function"""
    args = parse_cmd_args()
    results = pq.read_table(args.results).to_pandas()
    shard_costs = None
    if args.sizes is not None:
        shard_costs = pd.read_csv(args.sizes)
        assert len(shard_costs) == args.shards
        ndocs = shard_costs['shard_size'].sum()
        shard_costs['cost'] = shard_costs['shard_size'] * 100 / ndocs

    avg_p10, avg_mp = 0.0, 0.0
    if args.header:
        print('threshold,fold,p10,map,cost')
    for fold in range(args.folds):
        (p10, map_measure), avg_cost = eval_fold(
            results=results,
            qrels_path=args.qrels,
            fold_path=os.path.join(args.dir, f'fold-{fold}'),
            budget=args.budget,
            nshards=args.shards,
            nbuckets=args.buckets,
            shard_costs=shard_costs)
        avg_p10 += p10
        avg_mp += map_measure
        print(args.budget, fold, p10, map_measure, avg_cost, sep=',')
    avg_p10 /= args.folds
    avg_mp /= args.folds
    print('Avg. P@10: ', avg_p10, file=sys.stderr)
    print('Avg. MAP: ', avg_mp, file=sys.stderr)


main()
