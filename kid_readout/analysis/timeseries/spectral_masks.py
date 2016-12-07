import numpy as np

def pulse_tube_mask(frequency,S_xx,S_qq,S_qx,fundamental=1.4,max_harmonics=1,fractional_half_width=0.1,
                    min_bins_to_mask_per_harmonic=1):
    mask = np.ones(frequency.shape,dtype='bool')
    for harmonic in range(1,max_harmonics+1):
        center = fundamental*harmonic
        low = center*(1-fractional_half_width)
        high = center*(1+fractional_half_width)
        to_mask = (frequency>=low) & (frequency<=high)
        if to_mask.sum() < min_bins_to_mask_per_harmonic:
            center_idx = np.abs(frequency-center).argmin()
            min_idx = center_idx - min_bins_to_mask_per_harmonic//2
            if min_idx < 0:
                min_idx = 0
            max_idx = min_idx + min_bins_to_mask_per_harmonic
            if max_idx >= frequency.shape[0]:
                max_idx = frequency.shape[0]
            to_mask[min_idx:max_idx] = 1
        mask = mask & ~to_mask
    return mask
