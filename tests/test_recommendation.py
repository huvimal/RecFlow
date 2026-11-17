import unittest
import numpy as np
from recflow import split,fit,recommend,metrics

def fixture():
    return [dict(user=u,item=i,title=i+' HEART BAG',date='2011-01-01')
            for u,items in [('a',['1','2']),('b',['2','3']),('c',['3','4'])] for i in items]

class RecommendationTests(unittest.TestCase):
    def setUp(self):self.model=fit(fixture(),rank=2)
    def test_temporal_boundaries(self):
        events=[dict(date=d) for d in ['2011-08-31','2011-09-01','2011-10-15']]
        self.assertEqual([len(x) for x in split(events)],[1,1,1])
    def test_no_future_catalog_or_titles(self):
        events=fixture()+[dict(user='future',item='99',title='SECRET',date='2011-12-01')]
        m=fit(split(events)[0],rank=2)
        self.assertNotIn('99',m['items']);self.assertNotIn('secret',m['vocab'])
    def test_exclusion_for_all_methods(self):
        for method in ('collaborative','content','popularity'):
            _,rows=recommend(self.model,'a',10,method)
            self.assertTrue({'1','2'}.isdisjoint(r['product_id'] for r in rows))
    def test_cold_user_fallback_and_preference(self):
        strategy,rows=recommend(self.model,'new',2)
        self.assertIn('fallback',strategy);self.assertEqual(len(rows),2)
        strategy,_=recommend(self.model,'new',2,query='heart')
        self.assertIn('content',strategy)
        with self.assertRaises(ValueError):recommend(self.model,'new',query='zzzz')
    def test_metrics_include_unavailable_targets(self):
        recall,ndcg=metrics(['a'],{'a','unavailable'})
        self.assertEqual(recall,.5);self.assertLess(ndcg,1)
    def test_deterministic_training(self):
        other=fit(fixture(),rank=2)
        np.testing.assert_allclose(self.model['factors'],other['factors'])
        self.assertEqual(recommend(self.model,'a'),recommend(other,'a'))
    def test_small_catalog_and_invalid_inputs(self):
        self.assertLessEqual(len(recommend(self.model,'a',50)[1]),2)
        with self.assertRaises(ValueError):recommend(self.model,'a',0)
        with self.assertRaises(ValueError):recommend(self.model,'a',10,'unknown')
    def test_training_matrix_is_binary(self):
        duplicated=fit(fixture()+fixture(),rank=2)
        np.testing.assert_array_equal(self.model['seen'],duplicated['seen'])

if __name__=='__main__':unittest.main()
