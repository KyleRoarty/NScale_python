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

def ind_vals(arr, thresh, num_pts):
    out = []
    for i in range(num_pts):
        idx = np.argmax(arr)
        if arr[idx] < thresh:
            return out
        else:
            out.append(idx)
            arr[idx] = 0
    return out

def UC_location_corr(Data, N, num_preamble, DC, called_i, upsamp_factor, samp, overlap):
    upchirp_ind = []
    tmp_window = []
    DC_sum = sum(DC[0:N] * np.conj(DC[0:N]))
    for i in range(len(Data) - len(DC)):
        tmp_window.append(sum(Data[i:i+N] * DC[0:N]) /
            np.sqrt(sum(Data[i:i+N] * np.conj(Data[i:i+N])) *
            DC_sum))

    n_samp_array = []
    peak_ind_prev = []
    for i in range(math.floor(len(tmp_window)/N)):
        window = np.abs(tmp_window[i*N : (i+1)*N])
        peak_ind_curr = ind_vals(window, 0.2, 16)

        if len(peak_ind_prev) != 0 and len(peak_ind_curr) != 0:
            for j in range(len(peak_ind_curr)):
                for k in range(len(peak_ind_prev)):
                    if peak_ind_curr[j] == peak_ind_prev[k]:
                        n_samp_array.append(peak_ind_prev[k] + (i-1)*N)
        peak_ind_prev = peak_ind_curr

    for i in range(len(n_samp_array)):
        c = 0
        ind_arr = np.arange(0, (num_preamble-2)*N, N) + (n_samp_array[i] + N)

        for j in range(len(ind_arr)):
            c = c + np.sum(n_samp_array[:] == ind_arr[j])

        if c >= 6:
            upchirp_ind.append(n_samp_array[i])

    temp = []
    if len(upchirp_ind) != 0:
        temp.append(upchirp_ind[0])
        for i in range(1, len(upchirp_ind)):
            if (np.min(np.abs(upchirp_ind[i] - temp[:])) > 5):
                temp.append(upchirp_ind[i])
    if len(temp) != 0:
        return [int(idx * upsamp_factor + max(called_i*samp-overlap, 0)) for idx in temp]
    return temp

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
    raw_data = iof.read_iq('Packet_Collision_data_SF8/2_tx')

    ## Section 1.5: Find preambles
    upsampling_factor = Fs / BW
    N = 2**SF

    num_preamble = 8
    num_sync = 2
    preamble_sym = 1
    num_data_sym = config.Max_Payload_Num
    num_DC = 2.25
    pkt_len = num_preamble + num_DC + num_data_sym + num_sync
    num_samples = pkt_len * nsamp

    DC = sf.gen_normal(0, True, Fs)
    DC = DC.reshape((DC.size))

    chunks = 100
    overlap = num_preamble * nsamp
    samp = math.floor(len(raw_data)/chunks)
    preamble_ind = []

    global_start = time.time()

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    results = [pool.apply_async(UC_location_corr, args=(np.array(raw_data[max(int(i*samp - overlap), 0):int((i+1)*samp)])[::int(upsampling_factor)], N, num_preamble, DC[::int(upsampling_factor)], i, upsampling_factor, samp, overlap)) for i in range(chunks)]
    for r in results:
        ret = r.get()
        if len(ret) != 0:
            preamble_ind.extend(ret)

    print(f'Index processing done in {time.time() - global_start} seconds')

    for indice in preamble_ind:
        mdata = raw_data[indice - int(0.15 * nsamp) : indice + int(56.25 * nsamp)]
    #print(f'mdata: {len(mdata)}, {type(mdata)}')

    #frame_spectrum(mdata)

    # Start
        start_time = time.time()

        ## Section 2
        # Detect symbol in-window distribution
        num_wins = math.ceil(len(mdata) / nsamp)
        windows = [CWin(i) for i in range(num_wins)]

        # Pad mdata
        pad = np.zeros(int(nsamp*num_wins - len(mdata)), np.complex64)
        #print(f'Pad: {len(pad)}, {type(pad)}')
        mdata = np.append(mdata,pad)
        #print(len(mdata))

        symbs = [mdata[int(i*nsamp):int(i*nsamp+nsamp)] for i in range(num_wins)]


        results = [pool.apply_async(sf.detect, args=(symbs[i], )) for i in range(num_wins)]

        res = [r.get() for r in results]

        for i in range(num_wins):
            for s in res[i]:
                windows[i].addSymbol(s)
            if args.verbose:
                windows[i].show()

        ## Section 3
        # Detect LoRa frames by preambles
        start_win, bin_value = ff.detect(windows, args.verbose)

        if not start_win:
            print('ERROR: No packet is found!!!\n')
            return

        ## Section 4
        # Detect STO and CFO for each frame
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
        #num_pckts = len(pckts)
        #num_decoded = 0
        #offset = -1
        #num = 0
        #for pckt in pckts:
        #    print(f'Packet {num}:')
        #    message = lorad.lora_decoder(np.add(pckt, offset), SF)
        #    #check if decoded correctly
        #    if(not(message is None)):
        #        print(message)
        #        num_decoded = num_decoded + 1
        #        for bits in message:
        #            print(chr(int(bits)), end =" ")
        #    num = num + 1

        #print('')
        #print(f'{num_decoded} out of {num_pckts} decoded')

        end_time = time.time()
        print(f'Experiment finished in {end_time - start_time} seconds')

    global_end = time.time()
    print(f'Completely done in time  {global_end - global_start} seconds')

if __name__ == '__main__':
    main()
