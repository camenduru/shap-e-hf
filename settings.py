import os

import numpy as np

CACHE_EXAMPLES = os.getenv('CACHE_EXAMPLES') == '1'

MAX_SEED = np.iinfo(np.int32).max
