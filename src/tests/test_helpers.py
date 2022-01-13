import io

import numpy as np
import pytest
import scipy.signal

import helpers
import src.helpers


class TestLoadIntervals:
    interval_file_correct_testdata = [
        ("", []),
        ("c\t1\t2\n", [("c", 1, 2)]),
        ("c\t1\t3\nd\t10\t100\n", [("c", 1, 3), ("d", 10, 100)]),
    ]
    interval_file_incorrect_testdata = [
        "c\n",
        "c\t1\n",
        "c\t10\t2\n",
        "c\t10\t20\nc\t-2\t30\n",
    ]
    interval_file_with_empty_lines = [
        ("\n\n\n", []),
        ("\n\nc\t1\t2\n", [("c", 1, 2)]),
        ("c\t1\t3\n\n\n\nd\t10\t100\n\n\n\n\n", [("c", 1, 3), ("d", 10, 100)]),
    ]
    interval_file_closed = [
        ("c\t1\t10\n", [("c", 0, 10)]),
        ("c\t1\t10\n\nc2\t1\t1\n", [("c", 0, 10), ("c2", 0, 1)]),
    ]

    @pytest.mark.parametrize("text,expected", interval_file_correct_testdata)
    def test_load_intervals_correct(self, text, expected):
        result = src.helpers.load_intervals(io.StringIO(text))
        assert result == expected

    @pytest.mark.parametrize("text", interval_file_incorrect_testdata)
    def test_load_intervals_incorrect(self, text):
        with pytest.raises(ValueError):
            src.helpers.load_intervals(io.StringIO(text))

    @pytest.mark.parametrize("text,expected", interval_file_with_empty_lines)
    def test_load_interval_empty_lines_are_skipped(self, text, expected):
        result = src.helpers.load_intervals(io.StringIO(text))
        assert result == expected

    @pytest.mark.parametrize("text,expected", interval_file_closed)
    def test_load_intervals_correct(self, text, expected):
        result = src.helpers.load_intervals(io.StringIO(text), is_closed=True)
        assert result == expected


class TestLoadChrSizes:
    chr_sizes_file_correct_testdata = [
        ("", []),
        ("c\t12\n", [("c", 12)]),
    ]

    chr_sizes_file_incorrect_testdata = [
        "c\n",
        "c\t-2\n",
        "c\t2\t21\n",
        "c\t1\t10\n",
        "c\t1\t10\t2000\n",
    ]

    chr_sizes_file_empty_lines = [
        ("\n\n\n", []),
        ("\n\nc\t12\n\n\n\n", [("c", 12)]),
        ("\n\nc\t12\n\nc2\t33\n\n", [("c", 12), ("c2", 33)]),
    ]

    @pytest.mark.parametrize("text,expected", chr_sizes_file_correct_testdata)
    def test_load_chr_sizes_correct(self, text, expected):
        result = src.helpers.load_chr_sizes(io.StringIO(text))
        assert result == expected

    @pytest.mark.parametrize("text", chr_sizes_file_incorrect_testdata)
    def test_load_intervals_incorrect(self, text):
        with pytest.raises(ValueError):
            src.helpers.load_chr_sizes(io.StringIO(text))

    @pytest.mark.parametrize("text,expected", chr_sizes_file_empty_lines)
    def test_load_intervals_incorrect(self, text, expected):
        result = src.helpers.load_chr_sizes(io.StringIO(text))
        assert result == expected


class TestCountOverlaps:
    testdata = [
        ([("a", 1, 10)], [("a", 30, 40)], 0),
        ([("a", 1, 10)], [("a", 2, 15)], 1),
        ([("a", 1, 10)], [("a", 10, 15)], 0),
        ([("a", 1, 10)], [("b", 1, 10)], 0),
        ([("a", 1, 10)], [("a", 0, 1)], 0),
        ([("a", 1, 10)], [("a", 0, 1), ("a", 10, 12)], 0),
        ([("a", 0, 10)], [("a", 0, 1), ("a", 10, 12)], 1),
        ([("a", 0, 11)], [("a", 0, 1), ("a", 10, 12)], 1),
        ([("a", 0, 11), ("a", 12, 20)], [("a", 0, 1), ("a", 10, 12)], 1),
        ([("a", 0, 11), ("a", 12, 20)], [("a", 0, 1), ("a", 10, 12), ("a", 14, 20)], 2),
        ([("a", 0, 11), ("a", 12, 20), ("a", 70, 71)],
         [("a", 0, 1), ("a", 10, 12), ("a", 14, 20), ("a", 70, 71)], 3),
        ([("a", 0, 11), ("a", 12, 20), ("a", 70, 71)],
         [("a", 0, 1), ("a", 10, 12), ("a", 14, 20), ("a", 69, 71)], 3),
        ([("a", 0, 11), ("a", 12, 20), ("a", 70, 71)],
         [("a", 0, 1), ("a", 10, 12), ("a", 14, 20), ("a", 69, 72)], 3),
        ([("a", 0, 2), ("a", 10, 15), ("a", 16,18), ("a", 20, 21)],
         [("a", 0, 2), ("a", 9, 12), ("a", 17, 20), ("a", 69, 72)], 3),
    ]

    @pytest.mark.parametrize("r,q,expected", testdata)
    def test_count_overlaps(self, r, q, expected):
        result = src.helpers.count_overlaps(r, q)
        assert result == expected


class TestJointPvalue:
    testdata_empty_level = [
        ([[]]),
        ([[], [0, 1]]),
        ([[0, 1], [0, 2], []]),
    ]

    testdata_one_level = [
        ([0], 0, 1),
        ([0], 1, 0),
        ([np.log(0.5), np.log(0.5)], 0, 1),
        ([np.log(0.5), np.log(0.5)], 1, 0.5),
        ([np.log(0.5), np.log(0.5)], 2, 0),
        ([np.log(0.5), np.log(0.5)], 3, 0),
        ([-np.inf, np.log(0.5), np.log(0.5)], 0, 1),
        ([-np.inf, np.log(0.5), np.log(0.5)], 1, 1),
        ([-np.inf, np.log(0.5), np.log(0.5)], 2, 0.5),
        ([-np.inf, np.log(0.5), np.log(0.5)], 3, 0),
    ]

    testdata_multiple_levels = [
        ([[0], [0]], 0, 1),
        ([[0], [0]], 1, 0),
        ([[np.log(0.5), np.log(0.5)], [0]], 0, 1),
        ([[np.log(0.5), np.log(0.5)], [0]], -1, 1),
        ([[np.log(0.5), np.log(0.5)], [0]], 1, 0.5),
        ([[np.log(0.5), np.log(0.5)], [0]], 2, 0),
        ([[np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)]], 0, 1),
        ([[np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)]], 1, 0.75),
        ([[np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)]], 2, 0.25),
        ([[np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)]], 3, 0),
        ([[np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)]],
         -1, 1),
        ([[np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)]],
         0, 1),
        ([[np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)]],
         3, 1 / 8),
        ([[np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)]],
         2, 1 / 8 + 3 * 1 / 8),
        ([[np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)]],
         1, 1 / 8 + 3 * 1 / 8 + 3 * 1 / 8),
        ([[np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)], [np.log(0.5), np.log(0.5)],
          [np.log(0.5), np.log(0.5)]],
         4, 1 / 16),
        ([[np.log(1 / 3), np.log(1 / 3), np.log(1 / 3), ],
          [np.log(1 / 3), np.log(1 / 3), np.log(1 / 3), ]], 0, 1),
        ([[np.log(1 / 3), np.log(1 / 3), np.log(1 / 3), ],
          [np.log(1 / 3), np.log(1 / 3), np.log(1 / 3), ]], 1, 1 - 1 / 9),
        ([[np.log(1 / 3), np.log(1 / 3), np.log(1 / 3), ],
          [np.log(1 / 3), np.log(1 / 3), np.log(1 / 3), ]], 2, 1 - 3 / 9),
        ([[np.log(1 / 3), np.log(1 / 3), np.log(1 / 3), ],
          [np.log(1 / 3), np.log(1 / 3), np.log(1 / 3), ]], 3, 1 - 3 / 9 - 3 / 9),
        ([[np.log(1 / 3), np.log(1 / 3), np.log(1 / 3), ],
          [np.log(1 / 3), np.log(1 / 3), np.log(1 / 3), ]], 4, 1 / 9),
    ]

    def test_joint_pvalue_empty(self):
        with pytest.raises(ValueError):
            src.helpers.joint_pvalue([], 47)

    @pytest.mark.parametrize("level", testdata_empty_level)
    def test_joint_pvalue_each_level_should_be_nonempty(self, level):
        with pytest.raises(ValueError):
            src.helpers.joint_pvalue(level, 0)

    @pytest.mark.parametrize("level,k,expected", testdata_one_level)
    def test_joint_pvalue_one_level(self, level, k, expected):
        result = src.helpers.joint_pvalue([level], k)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("pvalues,k,expected", testdata_multiple_levels)
    def test_joint_pvalue_multiple_levels(self, pvalues, k, expected):
        result = src.helpers.joint_pvalue(pvalues, k)
        assert result == pytest.approx(expected)


class TestFFTConvolve:
    slow_testdata = [
        ([1], [1], [1]),
        ([1, 2], [1], [1, 2]),
        ([1, 2], [3, 4], [1*3, 1*4 + 2*3, 2*4]),
        ([1, 2, 3], [4, 5, 6], [1*4, 1*5 + 2*4, 1*6 + 2*5 + 3*4, 2*6 + 3*5, 3*6]),
        # ([1, 2, 3], [1]),
        # ([1, 2,], [1, 2]),
    ]

    slow_prob_merging_data = [
        ([1], [1], [1]),
        ([0.5, 0.5], [1], [0.5, 0.5]),
        ([0, 1], [0.5, 0.5], [0, 0.5, 0.5]),
        ([0.5, 0.5], [0.5, 0.5], [0.25, 0.5, 0.25]),
        ([1/3,1/3,1/3], [1/3,1/3,1/3], [1/9, 2/9, 3/9, 2/9, 1/9]),
        ([0,1/3,2/3], [1/3,1/3,1/3], [0/9, 1/9, 3/9, 3/9, 2/9])
    ]

    @pytest.mark.parametrize("a,b,expected", slow_testdata)
    def test_fft_convolve(self, a, b, expected):
        result = scipy.signal.fftconvolve(a,b)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("p,q,expected", slow_prob_merging_data)
    def test_fft_prob_merge(self, p, q, expected):
        result = scipy.signal.fftconvolve(p, q)
        assert result == pytest.approx(expected)


class TestSelectIntervalsByChrName:
    testdata = [
        ([], "a", []),
        ([("a", 0, 1)], "a", [(0, 1)]),
        ([("a", 0, 1), ("b", 1, 10)], "a", [(0, 1)]),
        ([("a", 0, 1), ("a", 2, 10), ("b", 1, 10)], "a", [(0, 1), (2, 10)]),
        ([("a", 0, 1), ("a", 2, 10), ("b", 1, 10)], "b", [(1, 10)]),
    ]

    @pytest.mark.parametrize("intervals,chr_name,expected", testdata)
    def test_select_intervals_by_chr_name(self, intervals, chr_name, expected):
        result = src.helpers.select_intervals_by_chr_name(intervals, chr_name)
        assert result == expected


testdata_merge_nondisjoint_intervals = [
    ([], []),
    ([("a", 0, 1), ("b", 0, 1)], [("a", 0, 1), ("b", 0, 1)]),
    ([("a", 0, 1), ("a", 0, 1)], [("a", 0, 1)]),
    ([("a", 0, 1), ("a", 0, 1), ("b", 2, 3)], [("a", 0, 1), ("b", 2, 3)]),
]


@pytest.mark.parametrize("intervals,expected", testdata_merge_nondisjoint_intervals)
def test_merge_nondisjoint_intervals(intervals, expected):
    result = helpers.merge_nondisjoint_intervals(intervals)
    assert result == expected
