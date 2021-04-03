#!/usr/bin/env python3

import config
import numpy as np
import peak_funcs as pf

from classes import CWin, CSymbol

def frame_spectrum(data, window=512, overlap=256, nfft=2048,
                   Fs=config.RX_Sampl_Rate):
    BW = config.LORA_BW
    SF = config.LORA_SF

    if Fs <= BW*2:
        window = 64
        overlap = 60
        nfft = 2048

def frame_detect(winset):
    Fs = config.RX_Sampl_Rate
    BW = config.LORA_BW
    SF = config.LORA_SF
    nsamp = Fs * 2**SF / BW

    start = []
    value = []

    state_dict = {}
    pending_keys = {}

    for i in range(0, len(winset)):
        print(f'window({i})')

        state_keys = list(state_dict)
        update_keys = {}
        print(f'Keys:', end='')
        for k in state_keys:
            update_keys[k] = 0
            print(f' {round(k)}', end='')
        print(f'\n', end='')


        symbset = winset[i].symset
        print(f'symbs:', end='')
        for k in symbset:
            print(f' {round(k.fft_bin)}', end='')
        print('\n', end='')

        for sym in symbset:
            # Detect consecutive preambles
            I, key = pf.peak_nearest(state_keys, sym.fft_bin, 2)
            if I < 0:
                state_dict[sym.fft_bin] = 1
            else:
                # Guaranteed to exist
                state_dict[key] += 1
                update_keys[key] = 1
                if state_dict[key] >= 5:
                    pending_keys[key] = 10

            # Detect the first sync word (8)
            I, key = pf.peak_nearest(state_keys, ((-1 + sym.fft_bin + 24) % 2**SF) + 1, 2)
            if I > 0 and key in pending_keys:
                print(f'SYNC-1: {round(key)}')
                pending_keys[key] = 10
                state_dict[key] += 1
                update_keys[key] = 1

            # Detect the second sync word (16)
            I, key = pf.peak_nearest(state_keys, ((-1 + sym.fft_bin + 32) % 2**SF) + 1, 2)

            # Short-circuits if second condition isn't true so never
            # have missing key exception
            if I > 0 and key in pending_keys and pending_keys[key] > 5:
                print(f'SYNC-2: {round(key)}\t Frame Detected')
                start.append(i-9)
                value.append(key)
                del pending_keys[key]
                update_keys[key] = 0

        for key in state_keys:
            if key in pending_keys and pending_keys[k] > 0:
                if update_keys[key] == 0:
                    pending_keys[key] -= 1
                    update_keys[key] = 1

            if update_keys[key] == 0:
                del state_dict[key]
                print(f'\tRemove {key:.2f} from table')

    return start, value
