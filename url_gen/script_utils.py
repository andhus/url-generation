from __future__ import print_function, division

import json
import os
import argparse

from datetime import datetime
from warnings import warn

DEFAULT_RESULTS_PATH = os.path.abspath(  # script outputs end up here.
    os.path.join(*([__file__] + [os.pardir] * 2 + ['results']))
)


def get_train_base_argument_parser(
    description=None,
    target_dir=DEFAULT_RESULTS_PATH,
    job_name=None,
    batch_size=32,
    epochs=1
):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--job-name',
        type=str,
        default=job_name,
        help='name of the training job')

    parser.add_argument(
        '-o', '--target-dir',
        type=str,
        default=target_dir,
        help='directory to write output to (model checkpoints etc.)')

    parser.add_argument(
        '--batch-size',
        type=int,
        default=batch_size,
        help='batch size to use in training')

    parser.add_argument(
        '--epochs',
        type=int,
        default=epochs,
        help='number of epochs to use in training')

    parser.add_argument(
        '--sample-fraction',
        type=float,
        default=None,
        help='use only a sub sample of data (useful for debugging)')

    return parser


def get_url_gen_argument_parser(
    units,
    embedding_size,
    url_cap_length,
    description=None,
    **kwargs
):
    """Helper method for argument parsing for MinHash -> LSH script."""
    parser = get_train_base_argument_parser(description, **kwargs)
    parser.add_argument(
        '--units',
        type=int,
        default=units,
        help='state size of the RNN')

    parser.add_argument(
        '--embedding-size',
        type=int,
        default=embedding_size,
        help='size of character embeddings')

    parser.add_argument(
        '--url-cap-length',
        type=int,
        default=url_cap_length,
        help='cap URL:s at this length during training')

    return parser


def check_get_versioned_job_path(job_path):
    while os.path.exists(job_path):
        warn('{} exists, appending version'.format(job_path))
        parts = job_path.split('#')
        if len(parts) > 1:
            try:
                version = int(parts[-1])
                version += 1
                parts[-1] = str(version)
            except ValueError:
                parts.append('1')
        else:
            parts.append('1')
        job_path = '#'.join(parts)

    return job_path


def initialize_job(args, append_time_job_name=False):
    if append_time_job_name:
        job_name = args.job_name + '_' + datetime.now().isoformat()
    else:
        job_name = args.job_name
    job_path = os.path.join(args.target_dir, job_name)
    job_path = check_get_versioned_job_path(job_path)
    mkdirp(job_path)
    arg_dict = vars(args)
    with open(os.path.join(job_path, 'arguments.json'), 'w') as f:
        json.dump(arg_dict, f)

    return job_path


def mkdirp(path):
    """Recursively creates directories to the specified path"""
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError('{} exists and is not a directory'.format(path))
    else:
        os.makedirs(path)
