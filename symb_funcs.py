#!/usr/bin/env python3

import config
import numpy as np

from classes import CSymbol
from scipy.signal import chirp
from scipy.fft import fft

def symb_detect(sig):
    Fs = config.RX_Sampl_Rate
    BW = config.LORA_BW
    SF = config.LORA_SF

    nsamp = Fs * 2**SF / BW
    MAX_PK_NUM = config.LORA_SF

    symbols = []
    threshold = 0

    for i in range(0,MAX_PK_NUM):
        # dechirping and fft
        dn_chp = symb_gen_normal(0, True)
        match_tone = np.multiply(sig, dn_chp)
        station_fout = fft(match_tone, int(nsamp*10))

        # applying non-stationary scaling down-chirp
        amp_lb = 1
        amp_ub = 1.2

        scal_func = np.linspace(amp_lb, amp_ub, int(nsamp))
        match_tone = np.multiply(sig, np.multiply(scal_func, dn_chp))
        non_station_fout = fft(match_tone, int(nsamp*10))

        # peak information
        pk_height = -1
        pk_index = 0
        pk_phase = 0

        # Iterative compensate phase rotation
        align_win_len = int(station_fout.size / (Fs/BW))

        pending_phase = np.arange(0, 20)*2*np.pi/20

        for p in pending_phase:
            non_scal_targ = np.exp(1j * p) * station_fout[:,0:align_win_len] +\
            station_fout[:,-align_win_len:]

            if np.max(np.absolute(non_scal_targ)) > pk_height:
                pk_index = np.argmax(np.absolute(non_scal_targ))
                pk_height = np.absolute(non_scal_targ)[:,pk_index]
                pk_phase = p

        # Threshold for peak detecting
        if i == 0:
            threshold = pk_height / 20
        else:
            if pk_height < threshold:
                break

        # Determin if peak is legitimate or duplicated
        repeat = False
        cbin = (1 - pk_index/align_win_len) * 2**SF

        for s in symbols:
            if np.absolute(s.fft_bin - cbin) < 2:
                repeat = True
                break

        if repeat:
            break

        # Scaling factor of peaks: alpha
        non_scal_targ = np.exp(1j * pk_phase) * non_station_fout[:,0:align_win_len] +\
        non_station_fout[:,-align_win_len:]

        alpha = np.absolute(non_scal_targ[:,pk_index]) / pk_height

        # Abnormal alpha
        if alpha < amp_lb or alpha > amp_ub:
            return

        freq = np.arange(0, align_win_len) * BW / align_win_len
        if alpha < (amp_lb + amp_ub) / 2:
            seg_len = (alpha - amp_lb) * 2 / (amp_ub - amp_lb) * nsamp
            amp = pk_height / seg_len
            dout, sym = symb_refine(True, seg_len[0], amp, freq[pk_index], sig)
        else:
            seg_len = (amp_ub - alpha) * 2 / (amp_ub - amp_lb) * nsamp
            amp = pk_height / seg_len
            dout, sym = symb_refine(False, seg_len[0], amp, freq[pk_index], sig)

        symbols.append(sym)
        sig = dout

    return symbols

def symb_gen_normal(code_word, down=False, Fs=config.RX_Sampl_Rate):
    BW = config.LORA_BW
    SF = config.LORA_SF

    org_Fs = Fs
    if Fs < BW:
        Fs = BW

    nsamp = Fs * 2**SF / BW

    T = np.arange(0, int(nsamp)) * 1/Fs

    f0 = -BW/2
    f1 = BW/2

    chirpI = chirp(T, f0, 2**SF/BW, f1, 'linear', 0)
    chirpQ = chirp(T, f0, 2**SF/BW, f1, 'linear', 90)

    baseline = chirpI + 1j * chirpQ

    if down:
        baseline = np.conjugate(baseline)

    baseline = np.tile(baseline, (1, 2))

    offset = round((2**SF - code_word) / 2**SF * nsamp)
    symb = baseline[:,offset:int(offset+nsamp)]

    if org_Fs != Fs:
        overSamp = int(Fs/org_Fs)
        symb = symb[:,0::overSamp]

    return symb

def symb_refine(near_prev, seg_length, seg_ampl, peak_freq, org_sig):
    Fs = config.RX_Sampl_Rate
    BW = config.LORA_BW
    SF = config.LORA_SF

    nsamp = Fs * 2**SF / BW

    min_residual = np.Inf
    dout = org_sig

    # Iteratively searching initial phase
    rphase_1 = 0
    rphase_2 = 0

    pending_phase = np.arange(0, 20)*2*np.pi/20

    for p in pending_phase:
        sig = seg_ampl * symb_gen_phase(2**SF * (1 - peak_freq/BW), p, rphase_2)
        if near_prev:
            sig[round(seg_length):] = 0
        else:
            sig[0: -round(seg_length)] = 0

        e_residual = np.sum(np.power(np.absolute(org_sig - sig), 2))
        if e_residual < min_residual:
            rphase_1 = p
            min_residual = e_residual
            dout = org_sig - sig

    for p in pending_phase:
        sig = seg_ampl * symb_gen_phase(2**SF * (1 - peak_freq/BW), rphase_1, p)
        if near_prev:
            sig[round(seg_length):] = 0
        else:
            sig[:-round(seg_length)] = 0

        e_residual = np.sum(np.power(np.absolute(org_sig - sig), 2))
        if e_residual < min_residual:
            rphase_2 = p
            min_residual = e_residual
            dout = org_sig - sig

    r_freq = peak_freq
    r_ampl = seg_ampl
    r_length = seg_length

    tmp = seg_ampl * (np.arange(0, int((1.1-0.9)/0.01)+1)*0.01 + 0.9)
    for i in tmp:
        sig - i * symb_gen_phase(2**SF * (1 - r_freq/BW), rphase_1, rphase_2)
        if near_prev:
            sig[round(seg_length):] = 0
        else:
            sig[:-round(seg_length)] = 0

        e_residual = np.sum(np.absolute(org_sig - sig))
        if e_residual < min_residual:
            r_ampl = i
            min_residual = e_residual
            dout = org_sig - sig

    tmp = np.linspace(seg_length - 50, seg_length + 50, int(100/5) + 1)
    for i in tmp:
        if i < 0 or i > nsamp:
            continue
        sig = r_ampl * symb_gen_phase(2**SF * (1 - r_freq/BW), rphase_1, rphase_2)
        if near_prev:
            sig[round(i):] = 0
        else:
            sig[:-round(i)] = 0

        e_residual = np.sum(np.absolute(org_sig - sig))
        if e_residual < min_residual:
            min_residual = e_residual
            dout = org_sig - sig
            r_length = i

    sym = CSymbol(near_prev, r_freq, r_ampl, r_length)

    return dout, sym

def symb_gen_phase(k, phase1, phase2, is_down = False):
    Fs = config.RX_Sampl_Rate
    BW = config.LORA_BW
    SF = config.LORA_SF

    nsamp = Fs * 2**SF / BW
    tsamp = np.arange(int(nsamp)) / Fs

    if is_down:
        f0 = BW/2
        f1 = -BW/2
    else:
        f0 = -BW/2
        f1 = BW/2

    chirpI = chirp(tsamp, f0, nsamp/Fs, f1, 'linear', 90)
    chirpQ = chirp(tsamp, f0, nsamp/Fs, f1, 'linear', 0)
    baseline = chirpI + 1j * chirpQ

    baseline = np.concatenate((np.multiply(baseline, np.exp(1j * phase1)),
                np.multiply(baseline, np.exp(1j * phase2))))

    offset = round((2**SF - k) / 2**SF * nsamp)
    return baseline[offset:int(offset+nsamp)]
