import numpy as np

ru = ['А', 'Б', 'Д', 'У', 'Р', 'Х', 'И', 'М', 'О', 'В', ' ', 'Ш', 'Т', 'Г', 'Л', 'Ч', 'К', 'Ж', 'Н', 'С', 'Е', 'З', 'Ё',
      'Я', 'Э', 'П', 'Й', 'Ф', 'Ы', 'Ь', 'Щ', 'Ю', 'Ц', '.', 'Ъ', "'", '-', 'І', '#']

en = [' ', "'", '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '`', '#']

ru_idx = dict(map(lambda x: (x[1], x[0]), enumerate(ru)))
en_idx = dict(map(lambda x: (x[1], x[0]), enumerate(en)))


class Vectorizer:
    def __init__(self, alphabet: dict):
        self.alphabet = alphabet
        self.length = len(alphabet)

    @classmethod
    def make_diff_alphabet(cls, alphabet):
        charset = ['', '--'] + list(alphabet.keys()) + ['+' + x for x in alphabet.keys()]
        return dict(map(lambda x: (x[1], x[0]), enumerate(charset)))

    def one_hot(self, char):
        vector = np.zeros(self.length, dtype=np.float32)
        char_idx = self.alphabet.get(char)
        if char_idx is not None:
            vector[char_idx] = 1.0
        return vector

    def label(self, char):
        return self.alphabet.get(char, -100)

    def vect_words(self, words: str or list, length=None, foo=None):
        if foo is None:
            foo = self.one_hot

        diff = (length or len(words)) - len(words)
        if isinstance(words, list):
            new_words = words + ['%'] * diff
        else:
            new_words = words + '%' * diff
        return np.stack([foo(x) for x in new_words])

    def vect_batch(self, batch, foo=None):
        if foo is None:
            foo = self.one_hot
        length = len(batch[0])
        res = np.stack([self.vect_words(x, length, foo=foo) for x in batch])
        lengths = list(map(len, batch))
        return res, lengths


if __name__ == '__main__':
    vect = Vectorizer(ru_idx)
    assert 'А' in ru_idx

    res, lengths = vect.vect_batch([
        "АРБД",
        'ПОГ'
    ])

    print(res.shape, lengths)
