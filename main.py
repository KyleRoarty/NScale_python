#!/usr/bin/env python3

import config
import io_funcs as iof
import math
import numpy as np
import symb_funcs as sf

from classes import CWin

def main():
    ## Section 1
    Fs = config.RX_Sampl_Rate
    BW = config.LORA_BW
    SF = config.LORA_SF

    nsamp = Fs * 2**SF / BW

    #Load raw signal
    mdata = iof.io_read_iq('input/collisions_2(-10dB)')
    #print(f'mdata: {len(mdata)}, {type(mdata)}')

    #frame_spectrum(mdata)

    ## Section 2
    # Detect symbol in-window distribution
    num_wins = math.ceil(len(mdata) / nsamp)
    windows = [CWin(i) for i in range(0, num_wins)]

    # Pad mdata
    pad = np.zeros(int(nsamp*num_wins - len(mdata)), np.complex64)
    #print(f'Pad: {len(pad)}, {type(pad)}')
    mdata = np.append(mdata,pad)
    #print(len(mdata))

    symbs = [mdata[int(i*nsamp):int(i*nsamp+nsamp)] for i in range(0, num_wins)]

    for i in range(0, num_wins):
        syms = sf.symb_detect(symbs[i])

        for s in syms:
            windows[i].addSymbol(s)

        windows[i].show()

if __name__ == '__main__':
    main()
