#!/usr/bin/env python3
import os
import gzip
import xml.etree.ElementTree as et
import io
import subprocess
from collections import Counter

import numpy as np
import ufal.morphodita as morphodita
import corpy.morphodita as corpy_m
import regex as re

from utils import Args
import pickle

class DocumentParser:
    # garbage = re.compile('[\.\,\"'\\\{\}\[\]/\+#@\$%\^\*&\(\)\<\>-_\?!]*')
    garbage_inside = re.compile('[\.\,\"\'\\\{\}\[\]/\+#@\$%\^\*&\(\)\<\>-_\?!]')
    def __init__(self, args, topic=False):
        self.args = args
        self.topic = topic

        self.morpho = None
        self.tokenizer = None
        self.tokenizer = corpy_m.Tokenizer(args.lang)

        if args.lang == 'czech' and args.terms == 'lemmas':
            self.morpho = morphodita.Morpho.load('./morpho.nosync/czech-morfflex-pdt-161115/czech-morfflex-161115.dict')
        elif args.lang == 'english' and args.terms == 'lemmas':
            self.morpho = morphodita.Morpho_load('./morpho.nosync/english-morphium-wsj-140407/english-morphium-140407.dict')

        if args.lang in ['czech', 'english']:
            file = 'language/{}-stops'.format(args.lang)
            with open(file, 'r') as f:
                self.stopwords = f.readlines()
        else:
            self.stopwords = []

    @classmethod
    def _try_get_item(cls, tag):
        """ Returns string of text in tag. """
        return tag.text if tag.text else ''

    @classmethod
    def _numbers(cls, word):
        # smartly replace numbers:
        # merge (-inf;-1]
        # merge [0;10]
        # merge [1900;2099]
        # merge [11;1899][2100;inf)
        word = re.sub('(10)|\d', 'numberten', word)
        word = re.sub('(20\d\d)|(19\d\d)', 'numberyear', word)
        word = re.sub('-\d+', 'numbernegative', word)
        word = re.sub('\d+', 'numberrest', word)

        return word

    @classmethod
    def _replace_with_lemma(cls, lemma):
        result = lemma
        if len(lemma) > 1:
            result = lemma.replace('_', '-').split('-')[0]
        return result

    @classmethod
    def _remove_interpunkt(cls, word):
        from_ = 'ěščřžýáíéůťďňú'
        to_ = 'escrzyaieutdnu'
        trantab = str.maketrans(from_, to_)
        return word.translate(trantab)

    @classmethod
    def _remove_garbage(cls, word):
        if len(word) <= 1:  # remove one char words
            return []
        if re.match('(\w+\d+\w*|\w*\d+\w+)', word):  # remove words with digits inside
            return []
        words = re.sub(cls.garbage_inside, ' ', word).split(' ')  #split words 
        return words

    @classmethod
    def _expand_q(cls, w):
        # take subsequences of length at least 3
        if len(w) <= 2:
            return [w]
        return [w[i:j] for i in range(len(w)-2) for j in range(max(3, i), len(w)+1) if i+3<=j]

    @classmethod
    def _smart_case(cls, w):
        # create This THIS and this from tHiS
        result = []
        result.append(w.lower())
        result.append(w.upper())
        if len(w) > 1:
            result.append(w.capitalize())
        return result


    def _extract_text(self, text, topic=False):
        """ Retruns list of forms in `conll_line_list` if `args.terms='forms'`
        if `args.terms='lemmas', list if lemmas.

        Tokens are lowercased if `args.lowercase` """

        if not self.args.train:
            items = list(self.tokenizer.tokenize(text))
        else:
            items = text


        if self.args.terms == 'lemmas' and not self.args.train:  # train stands for loading already lemmatized text
            lemmas = morphodita.TaggedLemmas()
            result = []
            for i,item in enumerate(items):
                self.morpho.analyze(item, self.morpho.GUESSER, lemmas)
                result.append(self._replace_with_lemma(list(lemmas)[0].lemma))

            # EXPAND QUERY
            if topic and self.args.expand_q:
                for item in items:
                    result.extend(self._expand_q(item))
            items = result
        elif self.args.terms == 'forms' and not self.args.train:
            # EXPAND QUERY when NO LEMMAS
            if topic and self.args.expand_q:
                result = []
                for item in items:
                    items.extend(self._expand_q(item))
                result = items

        # CASING QUERY
        if topic and self.args.lowercase_q:
            items = [w for x in items for w in self._smart_case(x)]  # TODO result or items ??? - split lemmas for query and for document

        # LOWERCASING DOCUMENT
        if not topic and self.args.lowercase_d:
            items = list(map(str.lower, items))

        # STOPS DELETION
        if self.args.del_stops:
            items = [x for x in items if x.lower() not in self.stopwords]

        # REMOVE INTERPUNCTION
        if self.args.rem_punkt:
            items = [self._remove_interpunkt(x) for x in items]

        # PROCESSING NUMBERS
        if self.args.process_numbers:
            items = [self._numbers(x) for x in items]

        # REMOVE GARBAGE
        if self.args.rem_garbage:
            items = [x for word in items for x in self._remove_garbage(word)]

        return items

    def _general_parse(self, file):
        try:
            with open(file, 'r') as f:
                tree = et.parse(f)
        except et.ParseError as e:
            # if the doc is not well-formed, fix it with xmllint
            pipe = subprocess.Popen(['xmllint', '--recover', file], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if pipe.returncode == 0:
                raise ValueError('DocumentParser: xmllint returned errors')
            text = re.sub('&\w*;? ?', '', pipe.communicate()[0].decode("utf-8"))
            with io.StringIO() as f:
                f.write(text)
                f.seek(0)
                tree = et.parse(f)

        return tree

    def _parse_topics(self, file):
        tree = self._general_parse(file)
        root = tree.getroot()
        tops = root.findall('top')
        doc_words_counts = {}
        for top in tops:
            resulting_text = []
            ident = None
            for tag in top:
                if tag.tag in self.args.query_fields:
                    text_in_tag = self._try_get_item(tag)
                    tokens_in_tag = self._extract_text(text_in_tag, True)  # processed text
                    resulting_text.extend(tokens_in_tag)
                elif tag.tag == 'num':
                    ident = tag.text.strip()

            doc_words_counts[ident] = np.unique(resulting_text, return_counts=True)

        return doc_words_counts


    def _parse_document(self, doc_path, words, doc_words_counts):
        tree = self._general_parse(doc_path)
        root = tree.getroot()
        docs = root.findall('DOC')
        prog = re.compile('(\n\r|\r|\n)')
        for doc in docs:
            resulting_text = []
            text = []
            for tag in doc:
                if tag.tag not in ['DOCNO', 'DOCID']:
                    text_in_tag = self._try_get_item(tag)
                    text.append(text_in_tag)
            text = re.sub(prog, ' ', ' '.join(text))
            tokens_in_tag = self._extract_text(text)  # processed text

            resulting_text.extend(tokens_in_tag)
            # update doc_counts for each word
            counts = Counter(resulting_text)
            for w, c in dict(counts).items():
                if not w in words:
                    words[w] = 0
                words[w] += c

            ident = doc.find('DOCNO').text.strip()
            doc_words_counts[ident] = np.unique(resulting_text, return_counts=True)


    def parse(self, list_path):
        """ Parse each document referenced in `list_path`.

        Args:
            list_path: path to file with list of paths of document

        Returns:
            (dict(words), dict(doc_words_counts)) where 
            the first dict containes word counts over all documents
            and the second, number of words for each documents
        """

        # IF NOT TRAINING
        file_path = os.path.join(self.args.data_prefix, list_path)
        doc_words_counts = {}
        words = {}
        with open(file_path, 'r') as document:
            for line in document:
                doc_path = os.path.join(self.args.data_prefix, self.args.document_path, line.strip())
                self._parse_document(doc_path, words, doc_words_counts)
        return words, doc_words_counts

    def parse_topics(self):
        file_path = os.path.join(self.args.data_prefix, self.args.q)
        return self._parse_topics(file_path)


if __name__=='__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', default='custom', type=str, help='id of the run')
    parser.add_argument('--pc', type=int, default=None, help='__ for development: pc ID')

    args_inline = parser.parse_args()

    # all arguments for the run
    args = Args(args_inline)
    doc_parser = DocumentParser(args)

    words, doc_words_counts = doc_parser.parse(args.d)



    # doc_parser = DocumentParser(args, True)
    # doc_words_counts = doc_parser.parse_topics()
