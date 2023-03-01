import numpy as np
from math import floor


def hist(window, n_bins):
    bins = np.linspace(window.min(), window.max(), n_bins)
    pts_per_bin, bin_edges = np.histogram(window, bins=bins, density=False)
    mode_bin = np.argmax(pts_per_bin)

    return pts_per_bin, bin_edges, mode_bin


def find_extrema_pause_onset(
    window,
    sample_offset,
    is_inhale,
    n_bins,
    upper_bins,
    lower_bins,
    min_bins_for_pause,
    signal_zero_cross,
):
    pts_per_bin, bin_edges, mode_bin = hist(window, n_bins)

    max_bin_ratio = pts_per_bin[mode_bin] / pts_per_bin.mean()

    is_pause = not (
        mode_bin < lower_bins
        or mode_bin > upper_bins
        or max_bin_ratio < min_bins_for_pause
    )

    if not is_pause:
        idx = np.where(
            (window <= signal_zero_cross if is_inhale else window > signal_zero_cross)
        )[0]

        extrema_onset = sample_offset + idx[-1]
        pause_onset = np.nan

    else:
        min_pause_range = bin_edges[mode_bin]
        max_pause_range = bin_edges[mode_bin + 1]
        max_pts_total = pts_per_bin[mode_bin]
        binning_thres = 0.25

        # Adds bins to the left
        for bin_added in range(1, max_pause_bins + 1):
            _bin = mode_bin - bin_added
            n_pts_added = pts_per_bin[_bin]
            if n_pts_added > max_pts_total * binning_thres:
                min_pause_range = bin_edges[_bin]

        # Adds bins to the right
        for bin_added in range(1, max_pause_bins + 1):
            _bin = mode_bin + bin_added
            n_pts_added = pts_per_bin[_bin]
            if n_pts_added > max_pts_total * binning_thres:
                max_pause_range = bin_edges[_bin]

        pause_idx = np.where((window > min_pause_range) & (window < max_pause_range))
        print(pause_idx)
        extrema_onset = sample_offset + pause_idx[-1] + 1
        pause_onset = sample_offset + pause_idx[0] - 1

    return extrema_onset, pause_onset


def find_onsets(y, peaks_idx, troughs_idx):
    n_bins = 100  # sr > 100 Hz

    max_pause_bins = 5 if n_bins >= 100 else 2

    min_bins_for_pause = 5
    upper_bins = round(n_bins * 0.7)
    lower_bins = round(n_bins * 0.3)
    signal_zero_cross = y.mean()

    inhale_onsets = np.empty(peaks_idx.shape).astype(int)
    inhale_pause_onsets = np.empty(peaks_idx.shape)
    inhale_pause_onsets[:] = np.nan
    exhale_onsets = np.empty(troughs_idx.shape).astype(int)
    exhale_pause_onsets = np.empty(troughs_idx.shape)
    exhale_pause_onsets[:] = np.nan

    # First inhale onset:
    avg_breath_dur = floor(
        np.array(
            [peaks_idx[i + 1] - peaks_idx[i] for i in range(len(peaks_idx) - 1)]
        ).mean()
    )

    first_zero_cross_boundary = (
        0 if peaks_idx[0] <= avg_breath_dur else peaks_idx[0] % avg_breath_dur
    )

    window = y[first_zero_cross_boundary : peaks_idx[0]]
    pts_per_bin, bin_edges, mode_bin = hist(window, n_bins)

    zero_cross_threshold = (
        signal_zero_cross
        if mode_bin < lower_bins or mode_bin > upper_bins
        else bin_edges[mode_bin]
    )

    first_potential_inhale = np.where(window < zero_cross_threshold)[0]
    inhale_onsets[0] = first_zero_cross_boundary + (
        first_potential_inhale[-1] if len(first_potential_inhale) > 0 else 0
    )

    # Onsets peak-peak
    for breath in range(len(peaks_idx) - 1):
        inhale_window = y[troughs_idx[breath] : peaks_idx[breath + 1]]
        (
            inhale_onsets[breath + 1],
            exhale_pause_onsets[breath],
        ) = find_extrema_pause_onset(
            inhale_window,
            troughs_idx[breath],
            True,
            n_bins,
            upper_bins,
            lower_bins,
            min_bins_for_pause,
            signal_zero_cross,
        )

        exhale_window = y[peaks_idx[breath] : troughs_idx[breath]]
        exhale_onsets[breath], inhale_pause_onsets[breath] = find_extrema_pause_onset(
            exhale_window,
            peaks_idx[breath],
            False,
            n_bins,
            upper_bins,
            lower_bins,
            min_bins_for_pause,
            signal_zero_cross,
        )

    # Last exhale onset:
    if y.shape[0] - peaks_idx[-1] > avg_breath_dur:
        last_zero_cross_boundary = peaks_idx[-1] + avg_breath_dur
    else:
        last_zero_cross_boundary = y.shape[0]

    exhale_window = y[peaks_idx[-1] : last_zero_cross_boundary]
    last_potential_exhale = np.where(exhale_window < signal_zero_cross)[0]

    exhale_onsets[-1] = (
        peaks_idx[-1] + last_potential_exhale[0]
        if len(last_potential_exhale) > 0
        else last_zero_cross_boundary
    )

    return inhale_onsets, exhale_onsets, inhale_pause_onsets, exhale_pause_onsets


def find_offsets(
    y, inhale_onsets, exhale_onsets, inhale_pause_onsets, exhale_pause_onsets
):
    inhale_offsets = np.zeros(inhale_onsets.shape)
    exhale_offsets = np.zeros(exhale_onsets.shape)

    for breath in range(len(exhale_onsets)):
        if np.isnan(inhale_pause_onsets[breath]):
            inhale_offsets[breath] = exhale_onsets[breath] - 1
        else:
            inhale_offsets[breath] = inhale_pause_onsets[breath] - 1

    for breath in range(len(exhale_onsets) - 1):
        if np.isnan(exhale_pause_onsets[breath]):
            exhale_offsets[breath] = inhale_onsets[breath + 1] - 1
        else:
            exhale_offsets[breath] = exhale_pause_onsets[breath] - 1

    final_window = y[exhale_onsets[-1] :]
    potential_exhale_offset = np.where(final_window > 0)[0]

    avg_exhale_dur = (exhale_offsets[:-1] - exhale_onsets[:-1]).mean()
    lower_lim = avg_exhale_dur / 4
    upper_lim = avg_exhale_dur * 1.75

    if (
        len(potential_exhale_offset) == 0
        or potential_exhale_offset[0] < lower_lim
        or potential_exhale_offset[0] > upper_lim
    ):
        exhale_offsets[-1] = np.nan
    else:
        exhale_offsets[-1] = exhale_onsets[-1] + potential_exhale_offset[0] - 1

    return inhale_offsets, exhale_offsets
