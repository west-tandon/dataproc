"""Cross-validation script for L2RR."""
import os
import re
import subprocess
import argparse
import numpy as np


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

def run_fold(feature_path, test_query_ids, output_dir):
    """Run a single fold."""
    test_query_ids = sorted(test_query_ids)
    print(f'Running a fold with test queries: {test_query_ids}')
    train_path, test_path = partition_data(
        feature_path, test_query_ids, output_dir)
    write_queries(test_query_ids, output_dir)
    model_path = os.path.join(output_dir, 'model')
    pred_path = os.path.join(output_dir, 'predictions')
    subprocess.run(f'svm_rank_learn -c 1 -t 0 {train_path} {model_path}'.split(' '))
    subprocess.run(f'svm_rank_classify {test_path} {model_path} {pred_path}'.split(' '))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run cross validation for L2RR on judged queries.')
    parser.add_argument('features', help='A file containing SVM-rank features.')
    parser.add_argument('output_dir', help='Output directory.')
    parser.add_argument('queries', help='An ordered list of queries to run.')
    parser.add_argument('--folds', '-k', default=10, type=int,
                        help='The number of folds to run.')
    args = parser.parse_args()
    with open(args.queries, 'r') as query_file:
        file_content = query_file.read()
        query_permutation = np.array([int(query) for query in file_content.split()])
    split = np.split(query_permutation, args.folds)
    for fold, test_queries in enumerate(split):
        run_fold(args.features,
                 test_queries,
                 os.path.join(args.output_dir, f'fold-{fold}'))

main()
