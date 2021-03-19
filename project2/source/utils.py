#!/usr/bin/env python3
import json
import os
from datetime import date

import numpy as np

class Funcs:
    term_weights = {\
                    'n': lambda data: data,\
                    'l': lambda data: 1 + safe_log(data),\
                    'L': lambda data: (1 + safe_log(data))/(1 + safe_log(np.mean(data))),\
                    'a': lambda data: 0.5 + (0.5*data)/np.max(data),\
                    'b': lambda data: np.ones_like(data, dtype=np.float32)\
                   }
    doc_weights = {\
                   'n': lambda N, df: 1, \
                   't': lambda N, df: log_frac(N, df), \
                   'p': lambda N, df: np.maximum(0, log_frac(N - df, df)) \
                  }
    norms = {\
             'n': lambda data, a, b: 1,\
             'c': lambda data, a, b: 1/np.linalg.norm(data),\
             'u': lambda data, piv, alpha: 1/((1 - alpha)*np.linalg.norm(data) + alpha*piv),\
             'b': lambda data, a, b: 1/0\
            }

    @classmethod
    def get_term_weight(cls, x, arg):
        """ Get weighted term frequency
        according to `arg` on x

        Args:
            arg: argument holding type of function
            x: array or number

        Returns:
            func(x) where `func` is given by `arg`

        Raises:
            ValueError: if value of `arg` has not been found
        """
        for (tw, func) in Funcs.term_weights.items():
            if arg == tw:
                res = func(x)
                return res
        raise ValueError('Funcs: \'arg\' has unknown value for term_weight: {}'.format(arg))

    @classmethod
    def get_doc_weight(cls, x, N, arg):
        """ Get weighted document frequency
        eccoring to `arg` on x

        Args:
            arg: argument holding type of function
            x: array or number

        Returns:
            func(x) where `func` is given by `arg`

        Raises:
            ValueError: if value of `arg` has not been found
        """
        for (dw, func) in Funcs.doc_weights.items():
            if arg == dw:
                res = func(N, x)
                return res
        raise ValueError('Funcs: \'arg\' has unknown value for doc_weight: {}'.format(arg))

    @classmethod
    def get_norm(cls, x, arg):
        """ Normalize `x` by `arg`

        Args:
            arg: norm type
            x: datapoint

        Returns:
            normalize datapoint `x`

        Raises:
            ValueError: if norm given by `arg` does not exist
        """
        for (no, func) in Funcs.norms.items():
            if arg == no:
                res = func(*x)
                return res
        raise ValueError('Funcs: \'arg\' has unknown value for norm: {}'.format(arg))



    @classmethod
    def get_func(cls, x, func):
        return func(x)

    @classmethod
    def get_func_on_csr(cls, x, arg):
        """ Apply function `func` on sparse matrix `x`.

        Args:
            func: function to apply on data of x
            x: a sparse matrix

        Returns:
            modified matrix x

        """
        return csr_matrix((func(x.data), x.indices, x.indptr), dtype=np.float32)


class Args:
    """ Consolidates all arguments of the experiment. """
    def __init__(self, args_inline, train=False, path=None):
        self._data = {}
        with open('args/args.json', 'r') as f:
            self._data.update(json.load(f))

        mapping = {'baseline':0, 'constrained':1, 'unconstrained':2, 'custom':'custom'}
        mapping_2 = {0: 'baseline', 1: 'constrained', 2: 'unconstrained'}
        langs = {'cs': 'czech', 'en': 'english'}
        # run-0_cs
        run = int(args_inline.r.split('-')[1].split('_')[0])
        args_inline.run = run
        args_inline.lang, args_inline.r = langs[args_inline.r.split('_')[-1]], mapping_2[run]
        if args_inline.run == 1 or args_inline.run == 2:
            run_args_path = 'args/args-{}-{}.json'.format(args_inline.run, args_inline.lang) if not train else path
        else:
            run_args_path = 'args/args-{}.json'.format(args_inline.run) if not train else path

        if os.path.isfile(run_args_path):
            with open(run_args_path, 'r') as f:
                self._data.update(json.load(f))
        self._data.update(vars(args_inline))

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)

    def __str__(self):
        return '\n'.join(['{}:  {}'.format(k,v) for k,v in self._data.items()])


def get_top_k(array, k=1000):
    """ Return indices of `k` highest values

    Args:
        array: array to select from
        k: how many indices to return

    Returns:
        Indices of `k` highest values

    """
    k = min(k, array.shape[1])
    # print(np.max(array, axis=1).shape)
    top_k = np.argpartition(array, -k, axis=1)[:, -k:]
    return top_k


def safe_log(a):
    return np.log10(a, where=a>0)


def log_frac(a, b):
    """ Returns log(a/b), more numerically stable. """
    return safe_log(a) - safe_log(b)
