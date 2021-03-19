#!/usr/bin/env python3
import os
import pickle
from itertools import combinations

import whoosh
import whoosh.index as WI
from whoosh.fields import *
from whoosh.qparser import QueryParser
import whoosh.qparser as WQ
from whoosh import scoring
import whoosh.analysis as WA
from whoosh.query import *
import corpy.morphodita as corpy_m
import numpy as np
import regex as re

from utils import Args

class MyTokenizer(WA.Tokenizer):
    def __init__(self, lang):
        super().__init__()
        self.tokenizer = PickableTokenizer(lang)
        self.token = WA.Token()
        self.token.positions=True

    def __call__(self, text, **kwargs):
        for i, token in enumerate(self.tokenizer.t.tokenize(text)):
            self.token.text = token
            self.token.pos = i
            yield self.token


class PickableTokenizer:
    def __init__(self, lang):
        self.lang =lang
        self.t = corpy_m.Tokenizer(lang)

    def __reduce__(self):
        return (PickableTokenizer, (self.lang,))


class PunctuationFilter(WA.Filter):
    def __init__(self):
        super().__init__()

    def __call__(self, tokens):
        for token in tokens:
            token.text = re.sub('\W', '', token.text)
            yield token

class NumbersFilter(WA.Filter):
    def __init__(self):
        super().__init__()

    def __call__(self, tokens, **kwargs):
        for token in tokens:
            token.text = re.sub('\d', '', token.text)
            yield token

class CleanupFilter(WA.Filter):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def __call__(self, tokens):
        for token in tokens:
            if len(token.text) > self.length:
                yield token

class RemoveCzechChars(WA.Filter):
    def __init__(self):
        super().__init__()
        from_ = 'ěščřžýáíéůťďňúüöä'
        to_ = 'escrzyaieutdnuuoa'
        self.trantab = str.maketrans(from_, to_)

    def __call__(self, tokens):
        for token in tokens:
            token.text = token.text.translate(self.trantab)
            yield token


class MyLemmaTokenizer(WA.Tokenizer):
    def __init__(self, path):
        super().__init__()
        self.tagger = PickableTagger(path)
        self.token = WA.Token()
        self.token.positions=False

    def __call__(self, text, **kwargs):
        for i, t in enumerate(self.tagger.t.tag(text, convert='strip_lemma_id')):
            self.token.text = t.lemma
            self.token.pos = i
            yield self.token


class PickableTagger:
    def __init__(self, path):
        self.path = path
        self.t = corpy_m.Tagger(path)

    def __reduce__(self):
        return (PickableTagger, (self.path,))



class Retrieval:
    def __init__(self, args):
        self.args = args
        self.ix_dir = 'model.nosync/index-{}-{}'.format(self.args.lang, self.args.r)
        create = False
        if os.path.exists(self.ix_dir):
            self.index = WI.open_dir(self.ix_dir)  # load index
            print(self.index.doc_count())
            print('loaded index')
        else:
            create = True

        if self.args.lang == 'czech':
            self.tagger_path = './morpho.nosync/czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger'
            self.imp = ['title', 'heading', 'text']
            with open('./language/czech-stops', 'r') as f:
                self.stopwords = map(str.strip, f.readlines())
        elif self.args.lang == 'english':
            self.tagger_path = './morpho.nosync/english-morphium-wsj-140407/english-morphium-wsj-140407.tagger'
            self.imp = ['title', 'heading', 'text']
            with open('./language/english-stops', 'r') as f:
                self.stopwords = map(str.strip, f.readlines())
        else:
            raise ValueError('wrong language')

        self.analyzer=self._get_analyzer()
        if create:
            self.index = self.create()  # create index



    def create(self):
        os.mkdir(self.ix_dir)
        with open('data.nosync/{}-new.pickle'.format(self.args.lang), 'rb') as f:
            docs = pickle.load(f)

        analyzer = self.analyzer

        schema = Schema(
            docno=ID(stored=True),
            text=TEXT(analyzer=analyzer, stored=True)
        )

        index = WI.create_in(self.ix_dir, schema)
        writer = index.writer(procs=4, multisegment=True, limitmb=512)

        for i, (docno, text) in enumerate(docs.items()):
            writer.add_document(docno=docno, text=text)
        writer.commit()
        return index


    def _get_analyzer(self):
        if self.args.run == 0:  # baseline
            final = WA.tokenizers.RegexTokenizer()
        else:  # constraint run (use only titles from queries)
            if self.args.lang == 'english':
                final = WA.analyzers.StemmingAnalyzer(stoplist=self.stopwords, cachesize=300000)
            elif self.args.lang =='czech':
                tokenizer = MyLemmaTokenizer(self.tagger_path)
                filterI = WA.LowercaseFilter() | \
                    WA.filters.StopFilter(self.stopwords) | \
                    RemoveCzechChars() | \
                    CleanupFilter(1)  # remove tokens t s.t.,  len(t) <= 1
                final = tokenizer | filterI
            else:
                raise ValueError('wrong lang')
        return final


    def _topics(self):
        with open('data.nosync/{}.pickle'.format(self.args.q.split('/')[-1]), 'rb') as f:
            topics = pickle.load(f)
        return topics


    def _parse_query(self, text, analyzer):
        items = []
        for p in processed:
            items.append(Term('content', p))
        return items


    def _score_topics_baseline(self, searcher):
        topics = self._topics()
        scores = [None] * len(topics)
        docs = [None] * len(topics)
        topic_idents = [None] * len(topics)

        for i, (ident, text) in enumerate(topics.items()):
            processed = [z.text for z in self.analyzer(text)]
            query = Or([Term('text', z) for z in processed])
            results = searcher.search(query, limit=1000)
            scores[i] = [results.score(n) for n in range(min(999, results.scored_length()))]
            docs[i] = [r['docno'] for r in results]  # docs number of i-th query
            topic_idents[i] = ident

        self.output(topic_idents, scores, docs)


    def _score_topics(self, searcher):
        topics = self._topics()
        scores = [None] * len(topics)
        docs = [None] * len(topics)
        topic_idents = [None] * len(topics)

        for i, (ident, text) in enumerate(topics.items()):
            processed = [z.text for z in self.analyzer(text)]
            query = Or([Term('text', z) for z in processed])
            # print(processed)

            results = searcher.search(query, limit=1000)


            # ========== pseudo relevance feedback ===========
            # Does not work  better than wihtout it
            # newtext = []
            # newtext.extend([word for word, score in results.key_terms('text', docs=5, numterms=6) if word not in processed])
            # print(newtext)
            # if len(newtext) > 0:
                # query2 = Or([Term('text', z, boost=0.7) for z in newtext])
                # query3 = Or([query, query2])
                # query3 = query3.normalize()
                # # query3 = query2
            # else:
                # query3 = query

            # results2 = searcher.search(query3, limit=900)
            # results.upgrade_and_extend(results2)
            # results = results2
            # print(results.scored_length())
            # print(results.estimated_length())

            scores[i] = [results.score(n) for n in range(min(999, results.scored_length()))]
            docs[i] = [r['docno'] for r in results]  # docs number of i-th query
            topic_idents[i] = ident

        self.output(topic_idents, scores, docs)


    def output(self, topic_idents, scores, docs):
        lines = []
        for i, topic in enumerate(topic_idents):
            for rank, (docno, score) in enumerate(zip(docs[i], scores[i])):
                lines.append('\t'.join([topic, '0', docno, str(rank), '{:.7f}'.format(score), self.args.r]))
        text = '\n'.join(lines)

        with open(self.args.o, 'w') as f:
            f.write(text)


    def search(self, B=None, K1=None):
        if B == None or K1 == None:
            if self.args.lang == 'english':
                B = 0.31
                K1 = 1.45
            elif self.args.lang == 'czech':
                B = 0.45
                K1 = 0.8
            else:
                raise ValueError('wrong language')
        if self.args.run == 0:  # baseline
            with self.index.searcher(weighting=scoring.Frequency) as searcher:  # only TF score
                self._score_topics_baseline(searcher)
        else:
            with self.index.searcher(weighting=scoring.BM25F(B=B, K1=K1)) as searcher:
                self._score_topics(searcher)


if __name__ == '__main__':
    import argparse
    from logger import *
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default='documents_cs.lst', type=str)
    parser.add_argument('-q', type=str, default='topics-train_cs.xml', help='Topics list file')
    parser.add_argument('-r', type=str, default='run-1_cs', help='Run name')
    parser.add_argument('-document_path', type=str, default='documents_cs', help='Run name')
    parser.add_argument('-f', type=int)
    parser.add_argument('-t', type=int)
    args_inline = parser.parse_args()


    args = Args(args_inline)
    # args._data['o'] = 'results.nosync/res_{}-{}-{}.res'.format(args.lang, args.r, str(9999999))
    if args.lang == 'english':
        qrels = 'A1/qrels-train_en.txt'
    else:
        qrels = 'A1/qrels-train_cs.txt'


    retrieval = Retrieval(args)
    retrieval.search(B=B, K1=K1)

    # THIS CODE IS ONLY FOR TRAINING

    tags = ['cs', 'lemma2', 'qexp3']
    # parameter search
    for experiment in range(args.f, args.t):
        logger = NeptuneLogger.new_experiment(tags)
        logger.log_status('started')


        #B = np.random.uniform(0.2, 1.0) 
        B = np.random.uniform(0.25, 0.43)
        K1 = np.random.uniform(1.0, 1.7)


        args._data['o'] = 'results.nosync/res_{}_{:.4f}-{:.4f}-{}.res'.format(args.lang, B, K1, '-'.join(tags))

        # SEARCH
        retrieval.search(B=B, K1=K1)
        logger.log_metrics({'B': B, 'K1':K1})

        pipe = subprocess.Popen(['A1/trec_eval-9.0.7/trec_eval', '-M1000', qrels, args.o], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        result_text = pipe.communicate()[0].decode("utf-8")
        logger.log_text('eval', result_text)

        # LOGGING
        lines = result_text.split('\n')
        inte = ['map', 'P_10']
        metrics = [l.strip() for l in lines if re.search('^({}) '.format('|'.join(inte)), l)]
        out = dict([(re.search('^[\w\d]+', l)[0], float(re.search('[\d\.]+$', l)[0])) for l in metrics])
        logger.log_metrics(out)
        logger.stop()

