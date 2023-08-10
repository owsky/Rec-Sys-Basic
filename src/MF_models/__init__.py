from .MF_base import MF_base
from .dense import MF_SVD_dense, MF_GD_dense
from .sparse import MF_SVD_sparse, MF_GD_sparse

GD_models_type = MF_GD_dense | MF_GD_sparse
SVD_models_type = MF_SVD_dense | MF_SVD_sparse
all_models_type = GD_models_type | SVD_models_type
