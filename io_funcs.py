#!/usr/bin/env python3

import numpy as np

def io_read_iq(filename, samples=-1):
    dat = np.fromfile(filename, np.complex64)
    return dat
