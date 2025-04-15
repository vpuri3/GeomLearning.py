from .masked import *
from .meshGNN import *

from .transolver import * # OG Transolver

# Unconditioned
from .ts1_uncond  import * # Slice attention1
from .ts2_uncond  import * # Slice attention2

# Conditioned
from .ts1 import * # Physics attention + AdaLN conditioning
from .ts2 import * # Slice attention + AdaLN conditioning (query size [M, D])
from .ts3 import * # Slice attention + slice query conditioning (query size [M, D])
from .ts4 import * # Slice attention + slice query conditioning (query size [H, M, D])
from .ts5 import * # Slice attention (full permute) + slice query conditioning (query size [H, M, D])
