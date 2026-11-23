import argparse
import os
import pickle

from utils.evaluation.covmat import CovMatEvaluator, print_covmat_results
from utils.misc import get_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='trusted local sampling results .pkl')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ratio', type=int, default=2)
    parser.add_argument('--start_idx', type=int, default=0)
    args = parser.parse_args()
    assert os.path.isfile(args.path)
    stem, _ = os.path.splitext(args.path)
    tag = os.path.basename(stem)
    logger = get_logger('eval', os.path.dirname(args.path), 'log_eval_' +
        tag + '.txt')
    logger.info('Loading results: %s' % args.path)
    with open(args.path, 'rb') as f:
        packed_dataset = pickle.load(f)
    logger.info('Total: %d' % len(packed_dataset))
    evaluator = CovMatEvaluator(num_workers=args.num_workers, ratio=args.
        ratio, print_fn=logger.info)
    results = evaluator(packed_data_list=list(packed_dataset), start_idx=
        args.start_idx)
    df = print_covmat_results(results, print_fn=logger.info)
    csv_fn = stem + '_covmat.csv'
    results_fn = stem + '_covmat.pkl'
    df.to_csv(csv_fn)
    with open(results_fn, 'wb') as f:
        pickle.dump(results, f)
