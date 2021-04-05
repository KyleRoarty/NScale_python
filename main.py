#!/usr/bin/env python3

import config
import io_funcs as iof
import math
import numpy as np
import os
import symb_funcs as sf
import frame_funcs as ff

from classes import CWin, CPacket

def main():
    ## Section 1
    Fs = config.RX_Sampl_Rate
    BW = config.LORA_BW
    SF = config.LORA_SF

    nsamp = Fs * 2**SF / BW

    #Load raw signal
    mdata = iof.read_iq('input/collisions_2(-10dB)')
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
        syms = sf.detect(symbs[i])

        for s in syms:
            windows[i].addSymbol(s)

        windows[i].show()

    ## Section 3
    # Detect LoRa frames by preambles
    start_win, bin_value = ff.detect(windows)

    if not start_win:
        print('ERROR: No packet is found!!!\n')
        return

    ## Section 4
    # Detect STO and CFO for each frame
    packet_set = [CPacket(0, 0, 0)] * len(start_win)
    for i in range(0, len(start_win)):
        print(f'y({start_win[i]}),value({bin_value[i]:.1f})')

        offset = round(bin_value[i] / 2**SF * nsamp)

        # Extra +1 because matlab was 1-indexed but python is 0-indexed
        upsig = mdata[(start_win[i]+2+1)*int(nsamp) + offset : (start_win[i]+2+1)*int(nsamp)+offset+int(nsamp)]
        downsig = mdata[(start_win[i]+9+1)*int(nsamp) + offset : (start_win[i]+9+1)*int(nsamp)+offset+int(nsamp)]

        cfo, sto = ff.cal_offset(upsig, downsig)
        sto = np.remainder(np.round(sto*Fs+offset+.25*nsamp), nsamp)
        packet_set[i] = CPacket(start_win[i], cfo, sto)
        print(f'Packet from {i}: CFO = {cfo:.2f}, TO = {sto}\n')

    ## Section 5
    # Group each symbol to corresponding TX

    # TODO: ensure outdir is created
    outfile = 'output/result.csv'
    if os.path.exists(outfile):
        os.remove(outfile)

    iof.write_text(outfile, f'{len(packet_set)}\n')
    iof.write_text(outfile, f'window,bin,offset,len,amplitude,belong,value')

    for w in windows:
        print(f'Window({w.ident})')

        symset = sf.group(w.symset, packet_set, w.ident)

        for s in symset:
            s.show()

            sto = nsamp - packet_set[s.pkt_id].to
            cfo = 2**SF * packet_set[s.pkt_id].cfo / BW

            value = np.mod(2**SF - s.fft_bin - sto/nsamp*2**SF - cfo, 2**SF)

            print(f'\t\t     value = {round(value)}')
            s.write_file(outfile, w.ident, s.pkt_id, round(value))

    ff.show(outfile)

    print('Experiment Finished!')

if __name__ == '__main__':
    main()
