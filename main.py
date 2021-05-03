#!/usr/bin/env python3

import sys
sys.path.append('C:\\Users\16085\\Desktop\\ECE901\\NScale_python')
import frame_funcs as ff
import argparse
import config
import io_funcs as iof
import math
import multiprocessing
import numpy as np
import os
import symb_funcs as sf
import time
import lora_decode_pyth as lorad

from classes import CWin, CPacket

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Print stuff')

    args = parser.parse_args()

    ## Section 1
    Fs = config.RX_Sampl_Rate
    BW = config.LORA_BW
    SF = config.LORA_SF

    nsamp = Fs * 2**SF / BW

    #Load raw signal
    #mdata = iof.read_iq('input/collisions_2(-10dB)')
    mdata = iof.read_iq('Packet_Collision_data_SF8//7_tx')
    mdata = mdata[435043:33655356]
    #print(f'mdata: {len(mdata)}, {type(mdata)}')

    #frame_spectrum(mdata)

    # Start
    start_time = time.time()

    ## Section 2
    # Detect symbol in-window distribution
    print("Section 2: Detect symbol in-window distribution")
    num_wins = math.ceil(len(mdata) / nsamp)
    windows = [CWin(i) for i in range(num_wins)]

    # Pad mdata
    pad = np.zeros(int(nsamp*num_wins - len(mdata)), np.complex64)
    #print(f'Pad: {len(pad)}, {type(pad)}')
    mdata = np.append(mdata,pad)
    #print(len(mdata))

    symbs = [mdata[int(i*nsamp):int(i*nsamp+nsamp)] for i in range(num_wins)]

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    results = [pool.apply_async(sf.detect, args=(symbs[i], )) for i in range(num_wins)]

    res = [r.get() for r in results]

    for i in range(num_wins):
        for s in res[i]:
            windows[i].addSymbol(s)
        if args.verbose:
            windows[i].show()

    ## Section 3
    # Detect LoRa frames by preambles
    print("Section 3: Detect LoRa frames by preambles")
    start_win, bin_value = ff.detect(windows, args.verbose)

    if not start_win:
        print('ERROR: No packet is found!!!\n')
        return

    ## Section 4
    # Detect STO and CFO for each frame
    print("Section 4: Detect STO and CFO for each frame")
    packet_set = [CPacket(0, 0, 0)] * len(start_win)

    results = [pool.apply_async(ff.cal_offset, args=(mdata[(start_win[i]+2+1)*int(nsamp) + round(bin_value[i] / 2**SF * nsamp) : (start_win[i]+2+1)*int(nsamp) + round(bin_value[i] / 2**SF * nsamp) + int(nsamp)], mdata[(start_win[i]+9+1)*int(nsamp) + round(bin_value[i] / 2**SF * nsamp) : (start_win[i]+9+1)*int(nsamp) + round(bin_value[i] / 2**SF * nsamp) + int(nsamp)])) for i in range(len(start_win))]

    res = [r.get() for r in results]

    for i in range(len(start_win)):
        if args.verbose == True:
            print(f'y({start_win[i]}),value({bin_value[i]:.1f})')

        offset = round(bin_value[i] / 2**SF * nsamp)

        sto = np.remainder(np.round(res[i][1]*Fs+offset+.25*nsamp), nsamp)
        packet_set[i] = CPacket(start_win[i], res[i][0], sto)
        if args.verbose == True:
            #print(f'Packet from {i}: CFO = {cfo:.2f}, TO = {sto}\n')
            print("above line causes error")

    ## Section 5
    # Group each symbol to corresponding TX
    print("Section 5: Group each symbol to corresponding TX")
    # TODO: ensure outdir is created
    outfile = 'output/result.csv'
    if os.path.exists(outfile):
        os.remove(outfile)

    iof.write_text(outfile, f'{len(packet_set)}\n')
    iof.write_text(outfile, f'window,bin,offset,len,amplitude,belong,value')

    results = [pool.apply_async(sf.group, args=(w.symset, packet_set, w.ident, args.verbose)) for w in windows]

    symsets = [r.get() for r in results]

    for i in range(len(windows)):
        if args.verbose == True:
            print(f'Window({windows[i].ident})')

        for s in symsets[i]:
            if args.verbose == True:
                s.show()

            sto = nsamp - packet_set[s.pkt_id].to
            cfo = 2**SF * packet_set[s.pkt_id].cfo / BW

            value = np.mod(2**SF - s.fft_bin - sto/nsamp*2**SF - cfo, 2**SF)

            if args.verbose == True:
                print(f'\t\t     value = {round(value)}')
            s.write_file(outfile, windows[i].ident, s.pkt_id, round(value))

    pckts = ff.show(outfile, True)
    
    ## Section 6 Decode Packets
    print("Section 6 Decode Packets")
    num_pckts = len(pckts)
    num_decoded = 0
    offset = 0
    num = 0
    for pckt in pckts:
        print(f'Packet {num}:')
        message = lorad.lora_decoder(np.add(pckt, offset), SF)
        #check if decoded correctly
        if(not(message is None)):
            print(message)
            num_decoded = num_decoded + 1
            for bits in message:
                print(chr(int(bits)), end =" ")
        num = num + 1
    
    print('')
    print(f'{num_decoded} out of {num_pckts} decoded')
        
    end_time = time.time()
    print(f'Experiment finished in {end_time - start_time} seconds')

if __name__ == '__main__':
    main()
