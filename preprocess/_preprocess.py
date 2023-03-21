import numpy as np
from math import floor
from sklearn.linear_model import LinearRegression
from scipy.fft import fft, ifft


def mean_smooth(y, sr):
    window_samples = int(sr / 1000 * 25)  # 25 ms windows

    y_padded = np.hstack(
        (
            np.zeros(
                window_samples - 1,
            ),
            y,
            np.zeros(
                window_samples,
            ),
        )
    )

    smoothed = np.zeros(
        y_padded.shape[0] - 5,
    )

    for i in range(y_padded.shape[0] - window_samples):
        if i < window_samples - 1:
            smoothed[i] = y_padded[i : i + window_samples].sum() / (i + 1)
        elif i > y_padded.shape[0] - 2 * window_samples:
            smoothed[i] = y_padded[i : i + window_samples].sum() / (
                y_padded.shape[0] - window_samples - i
            )
        else:
            smoothed[i] = y_padded[i : i + window_samples].mean()

    return smoothed


def find_global_drift(y):
    x = np.arange(0, y.shape[0]).reshape(-1, 1)
    return LinearRegression().fit(x, y).predict(x)


def remove_global_drift(y):
    return y - find_global_drift(y)


def find_local_drift(y, sr, period):
    w_size = floor(sr * period)

    # Build filter
    window = np.zeros((y.shape[0],))
    window[floor((y.shape[0] - w_size + 1) / 2) : floor((y.shape[0] + w_size) / 2)] = 1

    tmp = ifft(np.multiply(fft(y), fft(window)) / w_size)
    local_drift = -1 * ifft(np.multiply(fft(-1 * tmp), fft(window)) / w_size)

    return local_drift


def remove_local_drift(y, sr, period=60):
    return y - find_local_drift(y, sr, period)


def z_score(y):
    return (y - y.mean()) / y.std()
