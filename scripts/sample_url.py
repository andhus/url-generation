"""Generate phishy/non-phishy URL:s"""
from __future__ import print_function, division

import json
import os
import random

import six
import argparse


from url_gen.script_utils import DEFAULT_RESULTS_PATH


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--job-path',
        type=str,
        default=os.path.join(DEFAULT_RESULTS_PATH, 'DefaultJob'),
        help='The training job to load model from'
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
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
