from __future__ import print_function, division

import numpy as np

from url_gen.data import get_kaggle_urldataset

from keras import backend as K
from keras import layers
from keras.models import Model


if __name__ == '__main__':
    URL_MAX_LENGTH = 80
    LIMIT = None
    EMBEDDING_SIZE = 32
    RNN_UNITS = 512

    data = get_kaggle_urldataset()[:LIMIT]
    urls = data['url']
    phishing = data['phishing'].astype(float).values[:, None]

    pad_token = '<pad>'
    start_token = '<start>'
    end_token = '<end>'

    chars = [pad_token, start_token, end_token] + sorted(list(set(''.join(urls))))
    num_chars = len(chars)
    print('total chars:', num_chars)
    char_to_idx = dict((c, i) for i, c in enumerate(chars))
    idx_to_char = dict((i, c) for i, c in enumerate(chars))

    url_max_length_detected = urls.apply(len).max()  # TODO print
    max_length = min(URL_MAX_LENGTH, url_max_length_detected)

    url_idx = np.zeros((len(urls), max_length + 2), dtype=np.int)
    for i, url in enumerate(urls):
        for t, char in enumerate(
            [start_token] + list(url[:max_length]) + [end_token]
        ):
            url_idx[i, t] = char_to_idx[char]
    sequence_length = max_length + 1

    x = layers.Input((None, ), name='url_idx')
    c = layers.Input((1, ), name='phishing_label')
    initial_state = layers.Input((RNN_UNITS,), name='initial_state')

    embedding = layers.Embedding(num_chars, EMBEDDING_SIZE, mask_zero=True)(x)
    c_ext = layers.Lambda(
        lambda x_: K.repeat(c, K.shape(x)[1])
    )(c)
    embedding_c = layers.concatenate([embedding, c_ext], axis=-1)

    h1, state = layers.GRU(RNN_UNITS, return_sequences=True, return_state=True)(
        embedding_c, initial_state=initial_state
    )
    y_pred = layers.TimeDistributed(
        layers.Dense(num_chars, activation='softmax')
    )(h1)

    model = Model([x, c, initial_state], y_pred)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    model.fit(
        [url_idx[:, :-1], phishing, np.zeros((url_idx.shape[0], RNN_UNITS))],
        url_idx[:, 1:, None],
        epochs=3
    )

    model_with_state = Model([x, c, initial_state], [y_pred, state])

    def sample(hotstart='', phish_score=1, t_max=1e2):
        t = 0
        y_t = np.array([[char_to_idx[c] for c in [start_token] + list(hotstart)]])
        state_t = np.zeros((1, RNN_UNITS))
        generated_chars = []
        end_idx = char_to_idx[end_token]

        while y_t[0, -1] != end_idx and t < t_max:
            t += 1
            c_arr = np.array([[phish_score]])
            char_proba, state_t = model_with_state.predict([y_t, c_arr, state_t])
            next_char_idx = np.random.choice(np.arange(num_chars), p=char_proba[0, -1])
            generated_chars.append(idx_to_char[next_char_idx])
            y_t = np.array([[next_char_idx]])

        result = hotstart + ''.join(generated_chars)

        return result
