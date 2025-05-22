import unittest

import numpy as np

from pyunit_evaluate import topk


def get_edges(num: int):
    edges = []
    for e1 in range(0, num):
        for e2 in range(0, num):
            if e1 == e2:
                continue
            edges.append((e1, e2))

    node = np.array(edges, dtype=int)

    score = np.random.rand(node.shape[0], 1)
    value = np.concat([node, score], axis=1)
    return value


class TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        ## 数据格式
        ## 节点1  节点2 分数
        ## n1   n2  0.8
        ## n1   n3  0.7

        super(TestCase, self).__init__(*args, **kwargs)
        np.random.seed(22)
        predict = get_edges(20)
        self.predict = topk.Edges(predict)

        predict[:, 2] = np.random.randint(0, 2, predict.shape[0])
        self.true = topk.Edges(predict)

    def test_mrr(self):
        mr = topk.MRR(predict=self.predict, true=self.true)
        value = mr @ 5
        self.assertEqual(value, 0.4279)

    def test_map(self):
        ap = topk.MAP(predict=self.predict, true=self.true)
        value = ap @ 5
        self.assertEqual(value, 0.6413)

    def test_hits(self):
        hit = topk.Hits(predict=self.predict, true=self.true)
        value = hit @ 5
        self.assertEqual(value, 0.9)

    def test_precision(self):
        precisions = topk.Precision(predict=self.predict, true=self.true)
        value = precisions @ 5
        self.assertEqual(value, 0.47)

    def test_ndc(self):
        ndc = topk.NDCG(predict=self.predict, true=self.true)
        value = ndc @ 5
        self.assertEqual(value, 0.7807)

    def test_edges(self):
        es = topk.Edges()
        es.append(topk.Edge('e1', 'e2', 0.5))
        es.append(topk.Edge('e1', 'e3', 0.6))
        es.append(topk.Edge('e2', 'e3', 1))

        m2 = es.match(topk.Edge('e1', 'e2'))
        m1 = es.match(['e1', 'e2'])
        self.assertEqual(m1, m2)

        m3 = es.match(0)
        self.assertEqual(m3, es.items()[0])

        m4 = es.match('e1')
        self.assertEqual(len(m4), 2)


if __name__ == '__main__':
    unittest.main()
