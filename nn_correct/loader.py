from pprint import pprint

import Levenshtein
import csv
import re

from torch.utils.data import DataLoader

# cat train.csv | grep -E '.*,.*[A-Z].*,.*,[^2],.*' > train_en.csv
# cat train.csv | grep -vE '.*,.*[A-Z].*,.*,[^2],.*' > train_ru.csv
from nn_correct.vectorizer import Vectorizer, ru_idx


class FIOLoader(DataLoader):

    def __init__(self, file, batch_size, jobs, alphabet, sort_batch=True):
        self.alphabet = alphabet
        self.diff_alphabet = Vectorizer.make_diff_alphabet(alphabet)

        self.vectorizer = Vectorizer(alphabet)
        self.diff_vectorizer = Vectorizer(self.diff_alphabet)

        self.sort_batch = sort_batch
        self.jobs = jobs
        self.batch_size = batch_size
        self.file = file

    def read_batch(self):
        batch = []
        with open(self.file, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:

                if row['target'] == '0':
                    batch.append((row['country'], row['fullname'], row['fullname']))
                elif row['target'] == '1':
                    batch.append((row['country'], row['fullname'], row['fullname_true']))
                else:
                    pass

                if len(batch) >= self.batch_size:
                    if self.sort_batch:
                        batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
                    yield batch
                    batch = []

        if len(batch) > 0:
            if self.sort_batch:
                batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
            yield batch

    @classmethod
    def encode(cls, fullname, fullname_true):
        target = [''] * len(fullname)
        edit_opts = Levenshtein.editops(fullname, fullname_true)
        edit_opts = sorted(edit_opts, key=lambda x: (x[0], -x[1]), reverse=True)
        for op, src, dst in edit_opts:
            if op == 'delete':
                target[src] = '--'
            if op == 'replace':
                target[src] = fullname_true[dst]
            if op == 'insert':
                target[src] = '+' + fullname_true[dst]
        return target

    @classmethod
    def restore(cls, fullname, target):
        fullname = '#' + fullname + "#"
        res = []
        for src, tg in zip(fullname, target):
            if tg == '':
                res.append(src)
            elif tg == '--':
                pass
            elif len(tg) == 2 and tg[0] == '+':
                res.append(tg[1])
                res.append(src)
            else:
                res.append(tg)
        res = ''.join(res)
        return res.strip('#')

    def vectorize(self, batch):
        names = []
        diffs = []
        for country, fullname, fullname_true in batch:
            fullname = '#' + fullname + "#"
            fullname_true = '#' + fullname_true + "#"
            names.append(fullname)
            diff = self.encode(fullname, fullname_true)
            diffs.append(diff)

        name_mtx, lengths = self.vectorizer.vect_batch(names)
        diff_mtx, _ = self.diff_vectorizer.vect_batch(diffs, foo=self.diff_vectorizer.label)

        return name_mtx, lengths, diff_mtx

    def __iter__(self):
        for batch in self.read_batch():
            yield self.vectorize(batch)

    def __len__(self):
        return 1_000_000


if __name__ == '__main__':
    loader = FIOLoader('../train_ru.csv', 10, 1, alphabet=ru_idx)
    reader = loader.read_batch()

    batch1 = next(reader)

    [print(fio) for fio in batch1]

    name_mtx, lengths, diff_mtx = loader.vectorize(batch1)

    pprint(diff_mtx)
