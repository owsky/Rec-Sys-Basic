from collections import namedtuple

DefaultParams = namedtuple("DefaultParams", "n_factors epochs lr reg batch_size")

DenseStochasticParams = DefaultParams(10, 30, 0.01, 0.002, 1)
DenseBatchParams = DefaultParams(15, 50, 0.0009, 0.03, -1)
DenseMiniBatchParams = DefaultParams(12, 10, 0.01, 0.002, 32)
DenseSVDParams = DefaultParams(8, 20, 0.003, 0.04, 1)

SparseStochasticParams = DefaultParams(10, 30, 0.01, 0.002, 1)
SparseBatchParams = DefaultParams(15, 50, 0.0009, 0.03, -1)
SparseMiniBatchParams = DefaultParams(12, 10, 0.01, 0.002, 32)
SparseSVDParams = DefaultParams(8, 20, 0.003, 0.04, 1)
