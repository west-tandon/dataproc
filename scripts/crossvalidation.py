"""Cross-validation script for L2RR."""
import os
import re
import subprocess
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dataproc.data import scale_features, to_svmrank, load_data
from dataproc.selectivesearch import apply_cost_model


def parse_cmd_args():
    """Returns parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run cross validation for L2RR on judged queries.')
    parser.add_argument('data',
                        help='A CSV or Parquet file containing data.')
    parser.add_argument('output_dir', help='Output directory.')
    parser.add_argument('queries', help='An ordered list of queries to run.')
    parser.add_argument('--folds', '-k', default=10, type=int,
                        help='The number of folds to run.')
    parser.add_argument('--costs', help='A file containing a cost model.')
    parser.add_argument('--no-feature-scaling', action='store_true',
                        help='Do not scale features.')
    return parser.parse_args()


def partition_data(data_path, test_query_ids, output_dir):
    """Partition data set into train and test queries."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_path = os.path.join(output_dir, 'train.in')
    test_path = os.path.join(output_dir, 'test.in')
    qid_re = re.compile(r'.*qid:(?P<qid>[0-9]+)')
    with open(data_path, 'r') as data_file, \
            open(train_path, 'w') as train_file, \
            open(test_path, 'w') as test_file:
        for line in data_file:
            match = qid_re.match(line)
            qid = match.group('qid')
            if int(qid) in test_query_ids:
                test_file.write(line)
            else:
                train_file.write(line)
    return train_path, test_path


def write_queries(queries, output_dir):
    """Write a list of queries to a text file."""
    queries_path = os.path.join(output_dir, 'queries')
    with open(queries_path, 'w') as queries_file:
        for query in queries:
            print(query, file=queries_file)


def run_subprocess(cmd):
    """Run a subprocess defined by a string command."""
    return subprocess.run(cmd.split())


def run_fold(feature_path, test_query_ids, output_dir):
    """Run a single fold."""
    test_query_ids = sorted(test_query_ids)
    print(f'Running a fold with test queries: {test_query_ids}')
    train_path, test_path = partition_data(
        feature_path, test_query_ids, output_dir)
    write_queries(test_query_ids, output_dir)
    model_path = os.path.join(output_dir, 'model')
    pred_path = os.path.join(output_dir, 'predictions')
    run_subprocess(f'svm_rank_learn -c 1 -t 0 {train_path} {model_path}')
    run_subprocess(f'svm_rank_classify {test_path} {model_path} {pred_path}')


def create_svmrank_file(data, output_dir, cost_model_path):
    """Converts data to SVM-rank format, possibly applying a cost model."""
    if cost_model_path is not None:
        if cost_model_path.endswith('.csv'):
            cost_model = pd.read_csv(cost_model_path)
        else:
            cost_model = pq.read_table(cost_model_path).to_pandas()
        data = apply_cost_model(data, cost_model)
    out_path = os.path.join(output_dir, 'features.svmrank')
    to_svmrank(data.drop(columns=['shard']), out_path)
    return out_path


def main():
    """Main function"""
    args = parse_cmd_args()
    with open(args.queries, 'r') as query_file:
        file_content = query_file.read()
        query_permutation = np.array(
            [int(query) for query in file_content.split()])
    split = np.split(query_permutation, args.folds)
    data = load_data(args.data)
    if not args.no_feature_scaling:
        data, _ = scale_features(data, exclude=[
            'query', 'shard_score', 'shard', 'bucket'])
    feature_file = create_svmrank_file(data, args.output_dir, args.costs)
    for fold, test_queries in enumerate(split):
        run_fold(feature_file,
                 test_queries,
                 os.path.join(args.output_dir, f'fold-{fold}'))


main()
