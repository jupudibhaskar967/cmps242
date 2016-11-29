import json
import unittest

import constants
from dataset import Dataset, ReviewStat, DatasetStats

TEST_DATA = '{"votes": {"funny": 0, "useful": 0, "cool": 0}, ' \
            '"user_id": "PP_xoMSYlGr2pb67BbqBdA", ' \
            '"review_id": "7N9j5YbBHBW6qguE5DAeyA", "stars": 1, "date": "2014-10-29", ' \
            '"text": "Wing sauce is like water. Pretty much a lot of butter and some hot sauce (franks red hot maybe).  ' \
            'The whole wings are good size and crispy, but for $1 a wing the sauce could be better. ' \
            'The hot and extra hot are about the same flavor/heat.  ' \
            'The fish sandwich is good and is a large portion, sides are decent.", ' \
            '"type": "review", ' \
            '"business_id": "UsFtqoBl7naz8AVUBZMjQQ"}'
DATASET = Dataset('/home/karthik/PycharmProjects/cmps242/project/yelp_dataset_challenge_academic_dataset', 4, 100)


class DatasetTestCase(unittest.TestCase):
    def test_data_parsing(self):
        lines = [TEST_DATA]
        token_list = DATASET.process(lines)
        self.assertTrue(len(token_list) == 1)
        review_stat = token_list[0]
        self.assertEqual(self._get_review_text_from(TEST_DATA).sort(), review_stat.get_terms().sort())

    def test_review_stats(self):
        review_stat = self._build_review_stat()
        self.assertTrue(review_stat.has_term('the'))
        self.assertEqual(0.08333333333333333, review_stat.term_freq('the'))

    def test_dataset_stat(self):
        review_stat = self._build_review_stat()
        dataset_stat = DatasetStats()
        dataset_stat.add([review_stat])
        sorted_terms = dataset_stat.top_term_freq_prod_inv_doc_freq(1)
        for term, score in sorted_terms[:1]:
            self.assertEqual('extra', term)
            break

    def _build_review_stat(self):
        token_list = DATASET.parse(TEST_DATA)
        review_stat = ReviewStat()
        for token in token_list:
            review_stat.add(token)
        return review_stat

    def _get_review_text_from(self, data):
        text = json.loads(data).get(constants.TEXT, None)
        return DATASET.tokenize(DATASET.cleanse(text))


if __name__ == '__main__':
    unittest.main()
