import unittest
import numpy as np
from numpy.testing import assert_array_equal
from extrema_detection import find_corrected_extrema


class TestFindCorrectExtrema(unittest.TestCase):
    def test__start_with_troughs(self):
        single = np.array([0, 1, 0.9, 0, 0.1, 0.9, 1, 0.1, 0])
        potential_p_single = [1, 6]
        potential_t_single = [0, 3, 7]
        corrected_p_single, corrected_t_single = find_corrected_extrema(
            single, potential_p_single, potential_t_single
        )

        assert_array_equal(corrected_p_single, [1, 6], "Should keep all peaks")
        assert_array_equal(
            corrected_t_single, [3, 7], "Should remove first trough only"
        )

        multiple = np.array([0, 0.1, 0, 0.1, 1, 0.9, 0, 0.1, 0.9, 1, 0.1, 0])
        potential_p_multiple = [4, 9]
        potential_t_multiple = [0, 1, 2, 3, 6, 10]
        corrected_p_multiple, corrected_t_multiple = find_corrected_extrema(
            multiple, potential_p_multiple, potential_t_multiple
        )

        assert_array_equal(corrected_p_multiple, [4, 9], "Should keep all peaks")
        assert_array_equal(
            corrected_t_multiple, [6, 10], "Should remove all the first troughs only"
        )

    def test__2_consecutive_troughs(self):
        remove_trough = np.array([1, 0.9, 0, 0.1, 0.9, 1, 0.1, 0])
        potential_p = [0, 5]
        potential_t = [2, 3, 6, 7]
        corrected_p, corrected_t = find_corrected_extrema(
            remove_trough, potential_p, potential_t
        )

        assert_array_equal(corrected_p, [0, 5], "Should keep all peaks")
        assert_array_equal(
            corrected_t, [2, 6], "Should remove all the first troughs only"
        )

    def test__multiple_consecutive_troughs(self):
        remove_trough = np.array(
            [
                1,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                0.1,
                0,
            ]
        )
        potential_p = [0, 20]
        potential_t = [6, 7, 8, 9, 10, 11, 12, 13, 21]
        corrected_p, corrected_t = find_corrected_extrema(
            remove_trough, potential_p, potential_t
        )

        assert_array_equal(corrected_p, [0, 20], "Should keep all peaks")
        assert_array_equal(
            corrected_t, [10, 21], "Should keep one of consecutive troughs only"
        )

    def test__2_consecutive_peaks(self):
        remove_peak = np.array([1, 0.9, 0, 0.1, 0.9, 1, 0.1, 0])
        potential_p = [0, 1, 4, 5]
        potential_t = [2, 6]
        corrected_p, corrected_t = find_corrected_extrema(
            remove_peak, potential_p, potential_t
        )

        assert_array_equal(corrected_p, [0, 5], "Should remove consecutive peaks only")
        assert_array_equal(corrected_t, [2, 6], "Should keep all troughs")

    def test__multiple_consecutive_peaks(self):
        remove_peaks = np.array(
            [
                1,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                0.1,
                0,
            ]
        )
        potential_p = [0, 1, 2, 3, 4, 16, 17, 18, 19, 20]
        potential_t = [10, 21]
        corrected_p, corrected_t = find_corrected_extrema(
            remove_peaks, potential_p, potential_t
        )

        assert_array_equal(
            corrected_p, [0, 20], "Should keep one of the consecutive peaks only"
        )
        assert_array_equal(corrected_t, [10, 21], "Should keep all troughs")

    def test__all_removal(self):
        remove_all = np.array(
            [
                0,
                1,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                0.1,
                0,
            ]
        )
        potential_p = [1, 2, 3, 4, 5, 17, 18, 19, 20, 21]
        potential_t = [7, 8, 9, 10, 11, 12, 13, 14, 22]
        corrected_p, corrected_t = find_corrected_extrema(
            remove_all, potential_p, potential_t
        )

        assert_array_equal(corrected_p, [1, 21], "Should remove consecutive peaks")
        assert_array_equal(
            corrected_t, [11, 22], "Should remove first and consecutive troughs"
        )


if __name__ == "__main__":
    unittest.main()
