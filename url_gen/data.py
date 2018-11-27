from __future__ import print_function, division

import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(*([__file__] + [os.pardir] * 2))
)
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, 'datasets')


def drop_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def drop_www(url):
    return drop_prefix(url, 'www.')


def drop_http(url):
    return drop_prefix(url, 'http://')


def base_url_only(url):
    http_prefix = 'http://'
    if url.startswith(http_prefix):
        return http_prefix + url[len(http_prefix):].split('/')[0]
    return url.split('/')[0]


def get_kaggle_urldataset(
    url_preprocess=(
        drop_www,
        drop_http
    ),
    # url_filter=None  TODO add filtering of invalid URLs
):
    filepath = os.path.join(
        DEFAULT_DATA_PATH,
        'KaggleURLDataset',
        'urldata.csv'
    )
    df = pd.read_csv(filepath, encoding='utf8')
    df['phishing'] = df['label'] == 'bad'  # all are good or bad (no undefined)
    if url_preprocess is not None:
        if isinstance(url_preprocess, (tuple, list)):
            for ppf in url_preprocess:
                df['url'] = df['url'].apply(ppf)
        else:
            if not callable(url_preprocess):
                raise ValueError('')
            df['url'] = df['url'].apply(url_preprocess)

    return df[['url', 'phishing']]
