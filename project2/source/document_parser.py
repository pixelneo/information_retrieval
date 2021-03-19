#!/usr/bin/env python3
import os
import xml.etree.ElementTree as et
import io
import subprocess
import pickle

import numpy as np
import corpy.morphodita as corpy_m
import regex as re

from utils import Args

class DocumentParser:
    def __init__(self, args, topic=False):
        self.args = args
        self.topic = topic
        self.mapping = {'hd': 'title', 'ld':'heading', 'te':'text'}
        if self.args.lang == 'czech':
            # TITLE TEXT HEADING ELSE
            self.imp = ['title', 'heading', 'text']
        elif self.args.lang == 'english':
            # HD, LD, TE, else
            self.imp = ['hd', 'ld', 'te']
        else:
            raise ValueError('wrong lang')

    def _get_mapped(self, tagname):
        if tagname in self.imp:
            if tagname in self.mapping:
                return self.mapping[tagname]
            return tagname
        return 'rest'


    @classmethod
    def _try_get_item(cls, tag):
        """ Returns string of text in tag. """
        return tag.text if tag.text else ''


    def _general_parse(self, file):
        try:
            with open(file, 'r') as f:
                tree = et.parse(f)
        except et.ParseError as e:
            # if the doc is not well-formed, fix it with xmllint
            pipe = subprocess.Popen(['xmllint', '--recover', file], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if pipe.returncode == 1:
                raise ValueError('DocumentParser: xmllint returned errors')
            text = re.sub('&\w*;? ?', '', pipe.communicate()[0].decode("utf-8"))
            with io.StringIO() as f:
                f.write(text)
                f.seek(0)
                try:
                    tree = et.parse(f)
                except Exception as e:
                    print('------------------------- ERROR')
        return tree

    def _parse_topics(self, file):
        tree = self._general_parse(file)
        root = tree.getroot()
        tops = root.findall('top')
        result = {}
        for top in tops:
            resulting_text = []
            ident = None
            for tag in top:
                if tag.tag in ['title']:
                    text = self._try_get_item(tag)
                    resulting_text.append(text)
                elif tag.tag == 'num':
                    ident = tag.text.strip()
            result[ident] = ' '.join(resulting_text)
        return result


    def _parse_document(self, doc_path):
        """ Returns dict of resulting text. """
        tree = self._general_parse(doc_path)
        root = tree.getroot()
        docs = root.findall('DOC')
        prog = re.compile('(\n\r|\r|\n)')
        result = {}
        for doc in docs:
            text = []
            for tag in doc:
                if tag.tag not in ['DOCNO', 'DOCID']:
                    text_in_tag = self._try_get_item(tag)
                    text.append(text_in_tag)

            ident = doc.find('DOCNO').text.strip()
            result[ident] = ' '.join(text)
        return result


    def transform_docs(self):
        """ Only extraxt text from the files and dump it to pickled object. """
        file_path = os.path.join(self.args.data_prefix, self.args.d)
        dir_ = 0
        i = 0
        data = {}
        with open(file_path, 'r') as document:
            for line in document:
                doc_path = os.path.join(self.args.data_prefix, self.args.document_path, line.strip())
                docs = self._parse_document(doc_path)
                data.update(docs)
                docs.clear()
        # dump to pickle
        with open('data.nosync/{}-new.pickle'.format(self.args.lang.split('/')[-1]), 'wb') as f:
            pickle.dump(data, f)


    def transform_topics(self):
        """ Extract text from topic titles and dump it to pickle """
        file_path = os.path.join(self.args.data_prefix, self.args.q)
        data = self._parse_topics(file_path)
        # dump to pickle
        with open('data.nosync/{}.pickle'.format(self.args.q.split('/')[-1]), 'wb') as f:
            pickle.dump(data, f)


if __name__=='__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default='documents_cs.lst', type=str)
    parser.add_argument('-q', type=str, default='topics-train_cs.xml', help='Topics list file')
    parser.add_argument('-r', type=str, default='run-0_cs', help='Run name')
    parser.add_argument('-document_path', type=str, default='documents_cs', help='Run name')

    args_inline = parser.parse_args()

    # all arguments for the run
    args = Args(args_inline)
    doc_parser = DocumentParser(args)

    doc_parser.transform_docs()
    # doc_parser.transform_topics()
