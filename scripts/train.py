"""Train a Recurrent Neural Network to generate phishy and non-phishy URL:s"""
from __future__ import print_function, division

import os

import numpy as np

from url_gen.data import get_kaggle_urldataset
from url_gen.model import Charizer, get_gru_model, Sampler
from url_gen.script_utils import get_url_gen_argument_parser, initialize_job


if __name__ == '__main__':
    parser = get_url_gen_argument_parser(
        description=__doc__,
        job_name='DefaultJob',
        units=512,
        embedding_size=32,
        url_cap_length=1e2,
        batch_size=32,
        epochs=1,
    )
    args = parser.parse_args()
    print('Running training with args: {}'.format(args))
    job_path = initialize_job(args)

    print("Loading and preprocessing data...")
    data = get_kaggle_urldataset()
    urls = data['url']
    phishy = data['phishing'].astype(float).values[:, None]

    charizer = Charizer()
    charizer.fit_on_texts(urls)
    charizer.save(os.path.join(job_path, 'charizer'))
    url_idx = charizer.transform_texts(urls, cap_length=args)

    print('Creating model...')
    model_kwargs = {
        'num_chars': charizer.num_chars,
        'units': args.units,
        'embedding_size': args.embedding_size,
    }
    model = get_gru_model(returns_state=False, **model_kwargs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.fit(
        [url_idx[:, :-1], phishy, np.zeros((url_idx.shape[0], args.units))],
        url_idx[:, 1:, None],
        epochs=1
    )
    model_with_state = get_gru_model(returns_state=True, **model_kwargs)
    model_with_state.set_weights(model.get_weights())
    model_with_state.save_weights(os.path.join(job_path, 'model_weights.h5'))

    print('Done training!')
    sampler = Sampler(model_with_state, charizer, max_length=1e3)
    phishy_urls = [sampler.sample(phish_score=1.) for _ in range(10)]
    non_phishy_urls = [sampler.sample(phish_score=0.) for _ in range(10)]
    print("\n--- Phishy URL samples ---")
    for url in phishy_urls:
        print(url)

    print("\n--- Non-phishy URL samples ---")
    for url in non_phishy_urls:
        print(url)
