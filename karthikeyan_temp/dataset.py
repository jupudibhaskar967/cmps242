from __future__ import division, unicode_literals

import json
import math
import multiprocessing
import os

import constants


class Dataset(object):
    """Dataset class"""

    def __init__(self, dataset_dir, parallelism=4, batch_size=100):
        self._dataset_dir = dataset_dir
        self._pool = multiprocessing.Pool(parallelism)
        self._batch_size = batch_size
        self._dataset_stats = DatasetStats()

    def load(self):
        """
        Loads the data from the review dataset file.
        """
        review_data_file_path = os.path.join(self._dataset_dir, constants.REVIEW_DATA_FILE)
        result_list = []
        batches = []

        with open(review_data_file_path, 'r') as review_data_file:
            lines = []
            for line in review_data_file:
                if len(lines) == self._batch_size:
                    batches.append(lines)
                    lines = []
                else:
                    lines.append(line)
        # map the list of lines into a list of result dicts
        self._dataset_stats.add(self._pool.map(self.process, batches))
        self._pool.close()
        self._pool.join()
        return result_list

    def process(self, lines):
        review_stats = []
        for line in lines:
            tokens = self.parse(line)
            review_stat = ReviewStat()
            for token in tokens:
                review_stat.add(token)
            review_stats.append(review_stat)
        return review_stats

    def parse(self, line):
        """Parse the given line as json and extract the review text from the supplied lines"""
        data = json.loads(line)
        cleaned_review_text = self.cleanse(data.get(constants.TEXT, ""))
        return self.tokenize(cleaned_review_text)

    def _remove_punctuation(self, word):
        return ''.join(ch for ch in word if ch not in constants.PUNCTUATIONS)

    def cleanse(self, data):
        processed_data = self._remove_punctuation(data)
        processed_data = processed_data.lower()
        return processed_data

    def tokenize(self, line):
        return line.split()


class DatasetStats(object):
    def __init__(self):
        self._review_stats = []

    def add(self, review_stats):
        for review_stat in review_stats:
            self._review_stats.append(review_stat)

    def n_containing(self, term):
        return sum(1 for review_stat in self._review_stats if review_stat.has_term(term))

    def inverse_doc_freq(self, term):
        return math.log(len(self._review_stats) / (1 + self.n_containing(term)))

    def top_term_freq_prod_inv_doc_freq(self, count):
        scores = {}
        for review_stat in self._review_stats:
            scores = {term: self.term_freq_inv_doc_freq(term, review_stat) for term in review_stat.get_terms()}
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:count]

    def term_freq_inv_doc_freq(self, term, review_stat):
        return review_stat.term_freq(term) * self.inverse_doc_freq(term)


class ReviewStat(object):
    def __init__(self):
        self._term_count = {}
        self._total_words = 0

    def add(self, token):
        stripped_token = token.rstrip().lstrip()
        freq = self._term_count.get(stripped_token, 0)
        self._term_count[stripped_token] = freq + 1
        self._total_words += 1

    def get_terms(self):
        return self._term_count.keys()

    def has_term(self, term):
        return term in self._term_count

    def term_freq(self, term):
        return self._term_count.get(term, 0) / self._total_words
