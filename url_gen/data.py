from __future__ import print_function, division

import os
import subprocess

import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(*([__file__] + [os.pardir] * 2))
)
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, 'datasets')
SCRIPTS_PATH = os.path.join(PROJECT_ROOT, 'scripts')


def _require_dataset(name, filepath, script, auto_download=True):
    if not os.path.exists(filepath):
        script_path = os.path.join(SCRIPTS_PATH, script)
        if auto_download:
            process = subprocess.Popen(script_path, stdout=subprocess.PIPE)
            output, error = process.communicate()
            if error or not os.path.exists(filepath):
                raise IOError(
                    'Failed to auto-downloading {}, no such file or directly: {}. '
                    'Try to run: {} manually to download the data.'.format(
                        name, filepath, script_path)
                )
        else:
            raise IOError(
                'Failed to access {}, no such file or directly: {}. '
                'Try to run: {} to download the data.'.format(
                    name, filepath, script_path)
            )


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


def get_kaggle_urldata(
    url_preprocess=(
        drop_www,
        drop_http
    ),
    url_filter=None,
    auto_download=True,
):
    name = 'KaggleURLData'
    filepath = os.path.join(DEFAULT_DATA_PATH, name, 'urldata.csv')

    _require_dataset(
        name=name,
        filepath=filepath,
        script='download_kaggle_urldata.sh',
        auto_download=auto_download
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

    if url_filter is not None:
        raise NotImplementedError('filtering not implemented')  # TODO

    return df[['url', 'phishing']]
