from .logger import *
from .typing import *
from .utils import *

TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]

def get_batch_size(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.size(0)
    else:
        b_s = x[0].size(0)
    return b_s


def get_device(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.device
    else:
        b_s = x[0].device
    return b_s
