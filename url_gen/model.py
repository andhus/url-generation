from __future__ import print_function, division

import json
import os

import numpy as np

from keras import backend as K
from keras import layers
from keras.models import Model

from url_gen.script_utils import mkdirp


class Charizer(object):

    pad_token = '<pad>'  # always index zero
    start_token = '<start>'
    end_token = '<end>'

    def __init__(self, unique_chars=None):
        self.char_to_idx = None
        self.idx_to_char = None
        if unique_chars is not None:
            self._set_state(unique_chars)

    def _set_state(self, unique_chars):
        self.idx_to_char = [
            self.pad_token,
            self.start_token,
            self.end_token
        ] + sorted(list(unique_chars))
        self.char_to_idx = dict((c, i) for i, c in enumerate(self.idx_to_char))

    def fit_on_texts(self, texts):
        unique_chars = set([])
        for text in texts:
            unique_chars.update(text)
        self._set_state(unique_chars)
        return self

    def get_idx(self, char):
        return self.char_to_idx[char]

    def get_char(self, idx):
        return self.idx_to_char[idx]

    def transform_texts(self, texts, cap_length=None, pad_start=True, pad_end=True):

        max_lenght = max([len(t) for t in texts])
        text_length = min(cap_length, max_lenght) if cap_length else max_lenght
        out_length = text_length
        if pad_start:
            out_length += 1
        if pad_end:
            out_length += 1

        text_arr = np.zeros((len(texts), out_length), dtype=np.int)
        for i, text in enumerate(texts):
            if pad_start:
                text_arr[i, 0] = self.char_to_idx[self.start_token]
            for t, char in enumerate(text[:text_length]):
                text_arr[i, t + int(pad_start)] = self.char_to_idx[char]
            if pad_end:
                text_arr[i, t + int(pad_start) + 1] = self.char_to_idx[self.end_token]

        return text_arr

    @property
    def num_chars(self):
        if self.idx_to_char is None:
            raise RuntimeError(
                'Charizer must be fit before this property is defined'
            )
        return len(self.idx_to_char)

    def save(self, dirpath):
        mkdirp(dirpath)
        with open(os.path.join(dirpath, 'unique_chars.json'), 'w') as f:
            json.dump(sorted(self.idx_to_char[3:]), f)

    @classmethod
    def load(cls, dirpath):
        with open(os.path.join(dirpath, 'unique_chars.json')) as f:
            return cls(json.load(f))


def get_gru_model(
    num_chars,
    units,
    embedding_size=None,
    returns_state=False
):
    """

    :param num_chars:
    :param units:
    :param embedding_size:
    :return:
    """
    x = layers.Input((None,), name='url_idx')
    c = layers.Input((1,), name='phishing_label')
    initial_state = layers.Input((units,), name='initial_state')

    embedding = layers.Embedding(num_chars, embedding_size, mask_zero=True)(x)
    c_ext = layers.Lambda(
        lambda x_: K.repeat(c, K.shape(x)[1])
    )(c)
    embedding_c = layers.concatenate([embedding, c_ext], axis=-1)

    h1, state = layers.GRU(units, return_sequences=True, return_state=True)(
        embedding_c, initial_state=initial_state
    )
    y_pred = layers.TimeDistributed(
        layers.Dense(num_chars, activation='softmax')
    )(h1)

    if returns_state:
        model = Model([x, c, initial_state], [y_pred, state])
    else:
        model = Model([x, c, initial_state], y_pred)

    return model


class Sampler(object):

    def __init__(self, model, charizer, max_length=1e3):
        self.model = model
        self.charizer = charizer
        self.state_size = model.input_shape[-1][-1]
        # TODO only handles single state rnn:s
        self.max_length = max_length

    def _sample_idx(self, probabilities, temperature=1):
        p = np.exp(np.log(probabilities) / temperature)
        p = p / p.sum()
        return np.random.choice(np.arange(len(probabilities)), p=p)

    def sample(self, hotstart='', phish_score=1, temperature=1.):
        t = 0
        y_t = np.array([[
            self.charizer.get_idx(c)
            for c in [self.charizer.start_token] + list(hotstart)
        ]])
        state_t = np.zeros((1, self.state_size))
        generated_chars = []
        end_idx = self.charizer.get_idx(self.charizer.end_token)

        while y_t[0, -1] != end_idx and t < self.max_length:
            t += 1
            c_arr = np.array([[phish_score]])
            char_proba, state_t = self.model.predict([y_t, c_arr, state_t])
            next_char_idx = self._sample_idx(char_proba[0, -1], temperature)
            generated_chars.append(self.charizer.get_char(next_char_idx))
            y_t = np.array([[next_char_idx]])

        if generated_chars[-1] == self.charizer.end_token:
            generated_chars = generated_chars[:-1]
        result = hotstart + ''.join(generated_chars)

        return result
