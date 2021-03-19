#!/usr/bin/env python3
import os
import xml.etree.ElementTree as et
import pickle

import numpy as np

from document_parser import DocumentParser
from utils import Funcs
import sparse


class IncidenceMatrix:
    def __init__(self):
        self.words2ids = None # TODO unknown words
        self.docs2ids = None
        self.ids2docs = None
        self.dft = None
        self.matrix = None


    @classmethod
    def create(cls, args):
        """ Creates IM from list of documents in `args` """

        im = cls()

        parser = DocumentParser(args)
        vocab_dict, doc_words_counts = parser.parse(args.d)
        vocab_dict['<UNK>'] = 0
        vocab_sorted = np.sort(list(vocab_dict.keys()))

        # set mappings
        # im.ids2words = np.array(vocab_sorted)
        im.ids2docs = np.array(list(doc_words_counts.keys()))
        im.docs2ids = dict([(doc, i) for i, doc in enumerate(im.ids2docs)])
        im.words2ids = dict([(word, i) for i, word in enumerate(vocab_sorted)])
        im.dft = np.empty(len(im.words2ids))
        for i, w in enumerate(vocab_sorted):
            im.dft[i] = vocab_dict[w]

        ind_col = [] #np.arange(0, len(im.ids2docs))   # documents
        ind_row = [] #np.arange(0, len(im.ids2words))  # vocab
        data = []           # list of counts of words 
                            # matrix[ind_row[i], ind_col[i] = data[i]

        # Create matrix vocabulary x documents
        # term frequency, document frequency weight and normalization is applied here
        # doc_words_counts ={'id': [list_of_words, list_of_counts], ..}
        for doc, (words, counts) in doc_words_counts.items():
            # doc_term_f is actually `counts` -- ?? what
            doc_doc_f = np.empty_like(counts)
            for i, (word, count) in enumerate(zip(words, counts)):
                ind_col.append(im.docs2ids[doc])
                ind_row.append(im.words2ids[word])
                doc_doc_f[i] = im.dft[im.word2id(word)]

            tf = Funcs.get_term_weight(np.array(counts, dtype=np.float32), args.d_term_weight)
            df = Funcs.get_doc_weight(doc_doc_f, im.N, args.d_df_weight)
            weight = np.multiply(tf, df, dtype=np.float64)
            normalized = Funcs.get_norm((weight, args.d_vector_norm[1], args.d_vector_norm[2]), args.d_vector_norm[0])
            result = np.multiply(weight, normalized, dtype=np.float32)
            # resulting number after term freq, doc freq weighting and normalization
            data.extend(result)

        im.matrix = sparse.csc_matrix((np.array(data), (np.array(ind_row), np.array(ind_col))), \
                               shape=(len(im.words2ids), len(im.ids2docs)), dtype=np.float32)

        return im


    def word2id(self, word):
        """ Convert word to its ID or ID of <UNK>, if word is not present. """
        if word in self.words2ids.keys():
            return self.words2ids[word]
        return self.words2ids['<UNK>']


    @classmethod
    def load(cls, args):
        """ Loads already created IM. (in pickle file) """
        new = cls()
        file = os.path.join(args.train_folder, args.im_file)
        with open(file, 'rb') as f:
            new.words2ids,\
            new.docs2ids,\
            new.ids2docs,\
            new.dft,\
            new.matrix = pickle.load(f)
        return new

    def save(self, args):
        """ Save whole object to pickle file. """
        file = os.path.join(args.train_folder, args.im_file)
        with open(file, 'wb') as f:
            pickle.dump([self.words2ids, self.docs2ids, \
                         self.ids2docs, self.dft, self.matrix], f)

    @property
    def N(self):
        """ Number of documents. """
        return len(self.ids2docs)



if __name__ == '__main__':
    import argparse
    import os
    from utils import Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', default='custom', type=str, help='id of the run')
    parser.add_argument('--pc', type=int, default=None, help='__ for development: pc ID')
    args_inline = parser.parse_args()

    # all arguments for the run
    args = Args(args_inline)

    # im = IncidenceMatrix.load(args)
    im = IncidenceMatrix.create(args)
    im.save(args)

