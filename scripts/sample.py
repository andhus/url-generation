"""Sample phishy/non-phishy URL:s from a trained model."""
from __future__ import print_function, division

import json
import os
import random
import subprocess

import six
import argparse

from url_gen.script_utils import DEFAULT_RESULTS_PATH
from url_gen.data import SCRIPTS_PATH

DEFAULT_JOB_PATH = os.path.join(DEFAULT_RESULTS_PATH, 'DefaultJob')


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--job-path',
        type=str,
        default=DEFAULT_JOB_PATH,
        help='training job result directory to load the model from'
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.job_path):
        download = six.moves.input(
            'Could not find results for training job {} - '
            'do you want to download and sample from a '
            'default pretrained model? [y/n]: '.format(args.job_path)
        ).strip().lower()
        while download not in ('y', 'n'):
            overwrite = six.moves.input(
                'Enter "y" (download default model) or "n" (exit): '
            ).strip().lower()
        if download == 'n':
            exit(0)
        else:
            script_path = os.path.join(
                SCRIPTS_PATH,
                'download_default_job_result.sh'
            )
            process = subprocess.Popen(script_path, stdout=subprocess.PIPE)
            output, error = process.communicate()
            if error or not os.path.exists(args.job_path):
                print(
                    'Failed to download default model, please file an issue at '
                    'https://github.com/andhus/url-generation/issues'
                )
                exit(1)

    with open(os.path.join(args.job_path, 'arguments.json')) as f:
        job_args = json.load(f)

    from url_gen.model import get_gru_model, Charizer, Sampler

    charizer = Charizer.load(os.path.join(args.job_path, 'charizer'))
    model = get_gru_model(
        num_chars=charizer.num_chars,
        units=job_args['units'],
        embedding_size=job_args['embedding_size'],
        returns_state=True
    )
    model.load_weights(os.path.join(args.job_path, 'model_weights.h5'))
    sampler = Sampler(model=model, charizer=charizer, max_length=1e2)
    main_msg = (
        'Enter "<phishyness> <#samples> [<start of url>]" or just type "/" '
        'for random sample (type ":q" or press CTRL-C to exit)\n'
    )
    cmd = six.moves.input(main_msg)
    while True:
        if cmd == ':q':
            break
        if cmd == '/':
            ps = random.random()
            print(
                sampler.sample('', phish_score=ps),
                '[phishyness: {}]'.format(round(ps, 2))
            )
            cmd = six.moves.input('')
        else:
            try:
                cargs = cmd.split(' ')
                ps = float(cargs[0])
                num_samples = int(cargs[1])
                start = cargs[2] if len(cargs) > 2 else ''
                for _ in range(num_samples):
                    print(sampler.sample(hotstart=start, phish_score=ps))
                cmd = six.moves.input('')
            except:
                cmd = six.moves.input(main_msg)
