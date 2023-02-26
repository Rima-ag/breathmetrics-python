import numpy as np
from math import floor

SHIFT = 3
WINDOW_SIZES = [300, 500, 700, 1000, 5000]
def find_potential_extrema(y, window_sizes = WINDOW_SIZES, shift = SHIFT):
    troughs = []
    peaks = []

    for w_size in window_sizes:
        troughs.append([])
        peaks.append([])

        w_start = 0
        while w_start < y.size:
            window = y[w_start: min(w_start + w_size, len(y))]

            troughs[-1].append(np.argmin(window) + w_start)
            peaks[-1].append(np.argmax(window) + w_start)

            w_start += int(w_size / SHIFT)
        troughs[-1] = np.array(troughs[-1])
        peaks[-1] = np.array(peaks[-1])
    return peaks, troughs 


def find_threshold(votes, window_sizes = WINDOW_SIZES, shift=SHIFT):
    count_threshold = np.arange(1, len(window_sizes) * SHIFT)
    
    n_extrema = []
    
    for threshold in count_threshold:
        n_extrema.append((votes > threshold).sum())

    
    diff_n_extrema = []
    for n in range(1, len(n_extrema)):
        diff_n_extrema.append(n_extrema[n] - n_extrema[n - 1])

    imax = [i for i, j in enumerate(diff_n_extrema) if j == max(diff_n_extrema)][-1]
    
    return imax


def PWCT(peaks, troughs, window_sizes = WINDOW_SIZES, shift = SHIFT):
    _peaks = np.hstack(tuple(peaks)).flatten()
    _troughs = np.hstack(tuple(troughs)).flatten()

    peak_idx, peak_votes = np.unique(_peaks, return_counts=True)
    trough_idx, trough_votes = np.unique(_troughs, return_counts=True)
    
    peak_threshold = find_threshold(peak_votes, window_sizes, shift)
    trough_threshold = find_threshold(trough_votes, window_sizes, shift)

    best_decision_threshold = floor(np.array([peak_threshold, trough_threshold]).mean())
    
    return peak_idx[peak_votes >= best_decision_threshold], trough_idx[trough_votes >= best_decision_threshold]


def find_corrected_extrema(y, peaks_idx, troughs_idx):
    corrected_peaks = []
    corrected_troughs = []

    t_slow = 0
    while peaks_idx[0] > troughs_idx[t_slow]:
        t_slow += 1

    p_slow, p_fast = 0, 1
    t_fast = t_slow + 1

    while p_slow < len(peaks_idx) and t_slow < len(troughs_idx):
        if p_slow < len(peaks_idx) - 1:
            p_t_offset = troughs_idx[t_slow] - peaks_idx[p_slow]
            p_p_offset = peaks_idx[p_fast] - peaks_idx[p_slow]

            if p_p_offset < p_t_offset:
                fake_peak_idx = np.argmin([y[peaks_idx[p_slow]], y[peaks_idx[p_fast]]])
                
                if fake_peak_idx == 0:
                    p_slow += 1
                p_fast += 1
                
                continue

        if t_slow < len(troughs_idx) - 1:
            p_t_offset = troughs_idx[t_slow] - peaks_idx[p_slow]
            t_t_offset = troughs_idx[t_fast] - troughs_idx[t_slow]

            if t_t_offset < p_t_offset:
                fake_offset_idx = np.argmax([y[troughs_idx[t_slow]], y[troughs_idx[t_fast]]])
                
                if fake_offset_idx == 0:
                    t_slow += 1
                t_fast += 1
                
                continue

        p_t_offset = troughs_idx[t_slow] - peaks_idx[p_slow]
        assert p_t_offset > 0, f'Unexpected Error: peak found at {peaks_idx[p_slow]} after trough found at {troughs_idx[t_slow]},\n\
        Expected peaks to be followed by troughs.'

        corrected_peaks.append(peaks_idx[p_slow])
        corrected_troughs.append(troughs_idx[t_slow])
        
        p_slow = p_fast; p_fast += 1 
        t_slow = t_fast; t_fast += 1
        
    return corrected_peaks, corrected_troughs