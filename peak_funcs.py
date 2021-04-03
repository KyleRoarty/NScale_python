#!/usr/bin/env python3

import numpy as np

# Returns idx, value
def peak_nearest(arr, target, threshold):
    if not arr or np.isnan(target):
        return -1, -1

    min_idx = np.argmin(arr)

    if np.absolute(arr[min_idx] - target) > threshold:
        return -1, -1

    return min_idx, arr[min_idx]
