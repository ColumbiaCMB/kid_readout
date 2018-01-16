from glob import glob
from collections import OrderedDict

import numpy as np

file_pattern_180_mK = 'resonances_180_mK_????_MHz.npy'
band_list_180_mK = sorted([np.load(filename) for filename in glob(file_pattern_180_mK)], key=lambda b: b.min())
dict_180_mK = OrderedDict()
for band in band_list_180_mK:
    dict_180_mK['{:.0f}'.format(1e-6 * band.min())] = band

dict_maybe_220_GHz = OrderedDict([
    ('2881', 1e6 * np.array([2881.4, 2921.4, 2931.5, 2956.0, 2958.0, 2961.5, 2964.0])),
])

fake_resonances = 1e6 * np.array([
    1920, 2062, 2186, 2240, 2335, 2591, 2872, 3248, 3433, 3850, 3922
])

# New row means a gap of at least 20 MHz
real_resonances = 1e6 * np.array([
    2396,
    2676,
    2757,
    2778, 2792, 2805, 2806, 2816,
    2881, 2892,
    2921, 2931, 2944, 2946, 2955, 2958, 2961, 2965, 2980, 2982, 2995, 2998, 3001,
    3062, 3064, 3066, 3078, 3081, 3085, 3088, 3092, 3093, 3096, 3097, 3098, 3122, 3127, 3132, 3139, 3149,
    3169, 3188,
    3209, 3224, 3229,
    3251, 3263, 3268, 3277, 3278, 3279, 3293,
    3316, 3325, 3327,
    3347,
    3371,
    3398, 3405, 3410, 3415, 3421,
    3442, 3451, 3456, 3470
])