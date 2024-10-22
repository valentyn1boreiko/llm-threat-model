
from .autodan import AutoDan_adaptive
from .baseline import RedTeamingMethod
from .beast import BEAST_adaptive
from .gcg import GCG_adaptive
from .model_utils import _STR_DTYPE_TO_TORCH_DTYPE, _init_ray, load_model_and_tokenizer
from .pair import PAIR_adaptive

_method_mapping = {
    "BEAST": BEAST_adaptive,
    "GCG": GCG_adaptive,
    "PAIR": PAIR_adaptive,
    "AutoDan": AutoDan_adaptive,
    "AutoDan_adaptive": AutoDan_adaptive,
}

def get_method_class(method):
    """
    Returns the method class given the method name. This is used to access static methods.
    """
    if method not in _method_mapping:
        raise ValueError(f"Can not find method {method}")
    return _method_mapping[method]

def init_method(method, method_config):
    if method not in _method_mapping:
        raise ValueError(f"Can not find method {method}")
    method_class = _method_mapping[method]

    if method_class.use_ray:
        _init_ray(num_cpus=32)

    return method_class(**method_config)