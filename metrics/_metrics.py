import numpy as np

def find_time_between_breaths(sr, inhale_onsets):
    samples_between = np.array([inhale_onsets[i + 1] - inhale_onsets[i] for i in range(len(inhale_onsets) - 1)])
    time_between = samples_between / sr
    return time_between

def find_interbreath_interval(sr, inhale_onsets):
    time_between = find_time_between_breaths(sr, inhale_onsets)
    return time_between.mean()


def find_breathing_rate(inter_breath_interval):
    return 1 / inter_breath_interval


def find_coef_var_breathing_rate(sr, inhale_onsets):
    time_between = find_time_between_breaths(sr, inhale_onsets)
    return time_between.std() / time_between.mean()


def find_volumes(y, sr, inhale_onsets, inhale_offsets, exhale_onsets, exhale_offsets):
    inhale_volumes = np.zeros(inhale_onsets.shape)
    exhale_volumes = np.zeros(exhale_onsets.shape)

    for breath in range(inhale_onsets.shape[0]):
        if not np.isnan(inhale_offsets[breath]):
            inhale_volumes[breath] = abs(y[inhale_onsets[breath] : inhale_offsets[breath] + 1]).sum()
        else:
            inhale_volumes[breath] = np.nan

    for breath in range(exhale_onsets.shape[0]):
        if not np.isnan(exhale_offsets[breath]):
            exhale_volumes[breath] = abs(y[exhale_onsets[breath] : exhale_offsets[breath] + 1]).sum()
        else:
            exhale_volumes[breath] = np.nan

    inhale_volumes = inhale_volumes / sr * 1000
    exhale_volumes = exhale_volumes / sr * 1000

    return inhale_volumes, exhale_volumes


def find_coef_var_breath_volumes(inhale_volumes):
    return inhale_volumes.nanstd() / inhale_volumes.nanmean()


def find_tidal_volume(inhale_volumes, exhale_volumes):
    return inhale_volumes.nanmean() + exhale_volumes.nanmean()


def minute_ventilation(breathing_rate, tidal_volume):
    return breathing_rate * tidal_volume


def find_duration(sr, onsets, offsets):
    duration = np.zeros(onsets.shape)
    for breath in range(onsets.shape[0]):
        if not np.isnan(offsets[breath]):
            duration[breath] = offsets[breath] - onsets[breath]
        else:
            duration[breath] = np.nan


def find_duty_cycle(sr, onsets, offsets, interbreath_interval):
    duration = find_duration(sr, onsets, offsets)
    return duration.nanmean() / interbreath_interval


def find_coef_var_duty_cycle(sr, onsets, offsets):
    duration = find_duration(sr, onsets, offsets)
    return duration.nanstd() / duration.nanmean()