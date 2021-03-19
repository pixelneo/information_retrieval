#!/usr/bin/env python3

import numpy as np

import sparse
from incidence_matrix import IncidenceMatrix
from document_parser import DocumentParser
from utils import *

class Retrieval:
    def __init__(self, im, args):
        self.im = im
        self.args = args
        self.parser = DocumentParser(args, topic=True)
        self.change_topics()


    def change_topics(self):
        self.doc_words_counts = self.parser.parse_topics()
        self.ids2topics = np.array(list(self.doc_words_counts.keys()))
        self.topics2ids = dict(zip(self.ids2topics, np.arange(len(self.ids2topics))))


    def search(self):
        """ Run search of topics in documents

        Returns:
            (np.array(doc_names), np.array(scores)) of top 1000 matches

        """
        ind_col = []
        ind_row = []
        data = []

        for topic, (words, counts) in self.doc_words_counts.items():
            q_doc_f = np.array([self.im.dft[self.im.word2id(word)] for word in words], dtype=np.float32)
            for i, (word, count) in enumerate(zip(words, counts)):
                ind_row.append(self.topics2ids[topic])
                ind_col.append(self.im.word2id(word))
                # doc_doc_f[i] = im.dft[im.words2ids[word]]


            tf = Funcs.get_term_weight(np.array(counts, dtype=np.float32), self.args.q_term_weight)
            df = Funcs.get_doc_weight(q_doc_f, self.im.N, self.args.q_df_weight)
            weight = np.multiply(tf, df, dtype=np.float64)
            normalized = Funcs.get_norm((weight, self.args.q_vector_norm[1], self.args.q_vector_norm[2]), self.args.q_vector_norm[0])
            result = np.multiply(weight, normalized, dtype=np.float32)

            data.extend(result)

        matrix = sparse.csr_matrix((np.array(data), (np.array(ind_row), np.array(ind_col))), \
                            shape=(len(self.ids2topics), len(self.im.words2ids)), dtype=np.float32)

        scores = matrix.dot(self.im.matrix)

        # indices of top k documents, not sorted \in [0, ..., |docs|-1]
        top_k_ind = get_top_k(scores, self.args.top_k)

        # scores of top k documents, not sorted
        top_k_scores = np.take_along_axis(scores, top_k_ind, axis=1)

        # indices \in [0, ..., K-1] of sorted scores
        indices_of_sorted_scores = np.argsort(-top_k_scores, axis=1)

        # scores of top k docs, sorted
        top_k_scores = np.take_along_axis(top_k_scores, indices_of_sorted_scores, axis=1)

        # incides of top k docs, sorted \in [0, ..., |docs|-1]
        top_k_ind = np.take_along_axis(top_k_ind, indices_of_sorted_scores, axis=1)

        # top 1000 docs sorted decreasinglt by scores
        top_k_docs = self.im.ids2docs[top_k_ind]

        return top_k_docs, top_k_scores


    def perform_retrieval(self):
        """ Performs search for topics from input file given by -q argument
        and prints the result in file given by -o argument"""
        doc_names, scores = self.search()
        # 10.2452/401-AH 0 LN-20020201065 0 0 baseline
        # 1. qid 2. iter 3. docno 4. rank 5. sim 6. run_id
        lines = []
        for i, topic in enumerate(self.ids2topics):
            for rank, (docno, score) in enumerate(zip(doc_names[i], scores[i])):
                if score <= 0.0000001:
                    # score is too low
                    break
                lines.append('\t'.join([topic, '0', docno, str(rank), '{:.7f}'.format(score), self.args.r]))
        text = '\n'.join(lines)

        # note: o is argument of the program, given by assignement description
        with open(self.args.o, 'w') as f:
            f.write(text)

    def _check(self):
        print(self.topics2ids)


if __name__ == '__main__':
    import argparse
    import os
    from utils import Args
    import subprocess
    from logger import *
    import regex as re
    parser = argparse.ArgumentParser()
    parser.add_argument('--pc', type=int, default=None, help='__ for development: pc ID')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--tag', type=str, default=None, help='tag')
    parser.add_argument('-r', type=str, default='run-1_cs', help='Run name')
    parser.add_argument('-lang', type=str, default='english', help='Language')
    args_inline = parser.parse_args()

    args_inline = parser.parse_args()

    # all arguments for the run
    args = Args(args_inline, False, path=None)

    run_im_format = 'x'
    im_file = 'nic'
    tags = [args.lang] if args.tag is None else [args.lang, args.tag]
    for im_version in range(args_inline.start, args_inline.end+1):
        run_format = im_version
        for arg_file_path in os.listdir(os.path.join('args.nosync', str(im_version))):
            path = os.path.join('args.nosync', str(im_version), arg_file_path)
            args = Args(args_inline, True, path=path)
            logger = NeptuneLogger.new_experiment(tags, args)
            # logger.log_hyperparams(args)
            logger.log_status('started')
            with open(path, 'r') as f:
                logger.log_text('args', f.read())

            # run_im_format = '{}__{}-{}-{}-{}-{}-{}-{}-{}'.format(im_version,\
                                                 # args.d_term_weight,\
                                                 # args.d_df_weight,\
                                                 # args.d_vector_norm,\
                                                 # args.del_stops,\
                                                 # args.lowercase,\
                                                 # args.rem_punkt,\
                                                 # args.process_numbers,\
                                                 # args.terms)

            args._data['o'] = 'results.nosync/res_{}-{}.res'.format(run_format, arg_file_path.split('.')[0])
            args._data['r'] = 'run-1_cs'
            last_im_file = im_file
            im_file = 'im_{}.pickle'.format(run_format)
            args._data['im_file'] = im_file

            if os.path.exists(os.path.join('model.nosync',im_file)):
                im = IncidenceMatrix.load(args)
                print('loaded')
            else:
                if os.path.exists(os.path.join('model.nosync', last_im_file)):
                    os.remove(os.path.join('model.nosync', last_im_file))
                im = IncidenceMatrix.create(args)
                im.save(args)
                print('created and saved: {}'.format(im_file))

            retrieval = Retrieval(im, args)
            retrieval.perform_retrieval()

            # EVALUATING
            pipe = subprocess.Popen(['../A1/trec_eval-9.0.7/trec_eval', '-M1000', os.path.join(args.data_prefix, args.path_train_qrels), args.o], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            result_text = pipe.communicate()[0].decode("utf-8")
            logger.log_text('eval', result_text)

            # LOGGING
            lines = result_text.split('\n')
            inte = ['map', 'P_10']
            metrics = [l.strip() for l in lines if re.search('^({}) '.format('|'.join(inte)), l)]
            out = dict([(re.search('^[\w\d]+', l)[0], float(re.search('[\d\.]+$', l)[0])) for l in metrics])
            logger.log_metrics(out)
            logger.stop()
    try:
        os.remove(os.path.join('model.nosync', im_file))
    except:
        print('IM FILE NOT FOUND.')

