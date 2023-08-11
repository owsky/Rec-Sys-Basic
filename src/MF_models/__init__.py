from .MF_base import MF_base
from .MF_GD_all import MF_GD_all
from .MF_SVD_all import MF_SVD_all

all_models_type = MF_GD_all | MF_SVD_all
