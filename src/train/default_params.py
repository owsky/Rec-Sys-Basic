from collections import namedtuple

DefaultParams = namedtuple("DefaultParams", "n_factors epochs lr reg")
DefaultBatchParams = namedtuple(
    "DefaultBatchParams", "n_factors epochs lr reg batch_size"
)

DenseStochasticParams = DefaultParams(10, 30, 0.025, 0.009)
DenseBatchParams = DefaultParams(10, 30, 0.025, 0.009)
DenseMiniBatchParams = DefaultBatchParams(10, 20, 0.009, 0.002, 8)
DenseSVDParams = DefaultParams(10, 30, 0.0025, 0.009)

SparseStochasticParams = DefaultParams(10, 30, 0.025, 0.009)
SparseBatchParams = DefaultParams(10, 30, 0.025, 0.009)
SparseMiniBatchParams = DefaultBatchParams(10, 20, 0.009, 0.002, 8)
SparseSVDParams = DefaultParams(10, 30, 0.0025, 0.009)
