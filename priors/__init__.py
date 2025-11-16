# Import core priors that don't require botorch
from . import fast_gp, gp, ridge, stroke

# Try to import optional priors (may fail if dependencies not installed)
try:
    from . import fast_gp_mix
except ImportError:
    pass

try:
    from . import mlp
except ImportError:
    pass

try:
    from . import omniglot
except ImportError:
    pass

try:
    from . import binarized_regression
except ImportError:
    pass

try:
    from . import pyro
except ImportError:
    pass



