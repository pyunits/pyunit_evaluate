from abc import ABCMeta, abstractmethod
from typing import List

from .edges import Edges


class TopK(metaclass=ABCMeta):

    def __init__(self, predict: Edges, true: Edges, *, digits=4, exclude_zero=False):
        self.predict = predict
        self.true = true
        self.digits = digits
        self.exclude_zero = exclude_zero

    @abstractmethod
    def __matmul__(self, k: int) -> float:
        raise NotImplementedError

    def predict_group(self, k):
        for _, edges in self.predict.group('node1').items():
            if len(edges) > 1:
                edges = edges.sort(key=lambda x: x.score, reverse=True)

            valid = False
            for edge in edges:
                true = self.true.has_edge(edge)
                if true and true.score == 1:
                    valid = True
                    break

            if valid:
                yield edges[:k]

    def mean(self, scores: List[float]) -> float:
        if len(scores) == 0:
            return 0
        return round(sum(scores) / len(scores), ndigits=self.digits)
