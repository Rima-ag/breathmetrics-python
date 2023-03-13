import unittest
import numpy as np
from numpy.testing import assert_array_equal
from metrics import (
    find_interbreath_interval,
    find_breathing_rate,
    find_volumes,
    find_tidal_volume,
    find_minute_ventilation,
    find_duty_cycle,
    find_coef_var_breathing_rate,
    find_coef_var_breath_volumes,
    find_coef_var_duty_cycle,
)


class TestMetrics(unittest.TestCase):
    def test__find_interbreath_interval(self):
        inhale_onsets = np.array(
            [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
        )
        interbreath_interval = find_interbreath_interval(2, inhale_onsets)

        self.assertEqual(interbreath_interval, 500)

    def test__find_breathing_rate(self):
        inhale_onsets = np.array(
            [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
        )
        breathing_rate = find_breathing_rate(2, inhale_onsets)

        self.assertEqual(breathing_rate, 0.002)

    def test__find_coef_var_breathing_rate(self):
        inhale_onsets = np.array(
            [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
        )
        coef = find_coef_var_breathing_rate(2, inhale_onsets)

        self.assertTrue(coef < 1e-2)

        inhale_onsets = np.array(
            [0, 1000, 2200, 3000, 4500, 5700, 6800, 7900, 8100, 9200]
        )
        coef = find_coef_var_breathing_rate(2, inhale_onsets)

        self.assertTrue(coef > 3e-1)

    def test__find_volumes(self):
        inhale_onsets = np.array(
            [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
        )
        inhale_offsets = inhale_onsets + 250
        exhale_onsets = inhale_onsets + 500
        exhale_offsets = inhale_onsets + 750

        ones_250 = np.ones((250,))
        twos_250 = np.ones((250,)) * 2
        threes_250 = np.ones((250,)) * 3
        fours_250 = np.ones((250,)) * 4

        signal = np.array([])
        for i in range(10):
            signal = np.hstack((signal, ones_250, threes_250, twos_250, fours_250))

        inhale_volumes, exhale_volumes = find_volumes(
            signal, 2, inhale_onsets, inhale_offsets, exhale_onsets, exhale_offsets
        )

        assert_array_equal(
            inhale_volumes,
            np.array([(250 * 1 + 3) / 2 * 1000] * inhale_volumes.shape[0]),
        )
        assert_array_equal(
            exhale_volumes,
            np.array([(250 * 2 + 4) / 2 * 1000] * exhale_volumes.shape[0]),
        )

    def test__find_coef_var_breath_volumes(self):
        inhale_volumes = np.array([250, 250, 251, 250, 250, 250, 250, 250])
        coef = find_coef_var_breath_volumes(inhale_volumes)

        self.assertTrue(coef < 1e-2)

        inhale_volumes = np.array([500, 250, 251, 750, 250, 800, 250, 1000])
        coef = find_coef_var_breath_volumes(inhale_volumes)

        self.assertTrue(coef > 3e-1)

    def test__find_tidal_volume(self):
        inhale_volumes = [250, 250, 250, 750, 750, 750, np.nan]
        exhale_volumes = [np.nan, 200, 400, 200, 400, np.nan, 200, 400]

        tidal_volume = find_tidal_volume(inhale_volumes, exhale_volumes)
        self.assertEqual(tidal_volume, 500 + 300)

    def test__find_minute_ventilation(self):
        self.assertEqual(find_minute_ventilation(1, 1), 60)
        self.assertEqual(find_minute_ventilation(0.5, 0.2), 6)

    def test__find_duty_cycle(self):
        onsets = np.array([0, 300, 600, 900, 1200, 1500])
        offsets = np.hstack(([np.nan], onsets[1:-1] + 100, [np.nan]))
        duty_cycle = find_duty_cycle(
            2, onsets, offsets, find_interbreath_interval(2, onsets)
        )

        self.assertAlmostEqual(duty_cycle, 1 / 3)

    def test__find_coef_var_duty_cycle(self):
        onsets = np.array([0, 200, 400, 600, 800])
        offsets = np.array([100, 300, np.nan, 701, 902])
        coef = find_coef_var_duty_cycle(2, onsets, offsets)

        self.assertTrue(coef < 1e-2)

        onsets = np.array([0, 200, 400, 600, 800])
        offsets = np.array([100, 350, np.nan, 670, 902])
        coef = find_coef_var_duty_cycle(2, onsets, offsets)
        self.assertTrue(coef > 25e-2)
