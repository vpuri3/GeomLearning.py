from .masked import *
from .meshGNN import *

from .transolver import * # OG Transolver

# Unconditioned
from .ts1_uncond  import * # Slice attention1
from .ts2_uncond  import * # Slice attention2

# Conditioned
from .ts1 import * # Physics attention + AdaLN conditioning
from .ts2 import * # Slice attention + AdaLN conditioning
from .ts3 import * # Slice attention (+ temperature) + AdaLN conditioning
from .ts4 import * # Slice attention (+ temperature) + slice query conditioning
#