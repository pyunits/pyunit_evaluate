from .topk import TopK


class MRR(TopK):
    """
    MRR(Mean Reciprocal Rank)是链路预测中常用的评估指标之一，用于衡量模型预测的准确性。
    MRR是"平均倒数排名"的缩写，它通过计算预测结果中正确链接的排名的倒数来衡量模型性能。在链路预测中，MRR特别适用于评估模型对缺失链接的排序质量。

    MRR的计算步骤
    1. 对每个测试集中的正样本
    2. 确定正确边的排名
    3. 计算倒数排名
    4. 计算所有测试边的平均倒数排名

    """

    def __matmul__(self, k: int) -> float:
        values = []

        for edges in self.predict_group(k):
            rank = []
            for idx, item in enumerate(edges.items(), start=1):
                edge = self.true.has_edge(item)
                if edge and edge.score == 1:
                    rank.append(1 / idx)

            if self.exclude_zero and len(rank) == 0:
                continue

            values.append(self.mean(rank))

        return self.mean(values)
