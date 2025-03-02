from .masked import *
from .meshGNN import *

from .transolver import * # OG Transolver
from .ts  import * # Slice attention + no conditioning
from .ts1 import * # Physics attention + AdaLN conditioning
from .ts2 import * # Slice attention + AdaLN conditioning
from .ts3 import * # Slice attention (+ temperature) + AdaLN conditioning
from .ts4 import * # Slice attention (+ temperature) + slice query conditioning
#