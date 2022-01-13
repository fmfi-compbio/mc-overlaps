import datetime
import itertools

import numpy as np
import pytest
import scipy.stats
from scipy.special import logsumexp

import simple_model
import src.simple_model


class TestGenerateEmission:
    def test_number_of_states_is_correct(self):
        t = np.array([[1, 0], [0, 1]], dtype=np.float)
        length = 10
        states = src.simple_model.Model.generate_emission(t, length)
        assert len(states) == length

    def test_starts_with_zero(self):
        t = np.array([[1, 0], [0, 1]], dtype=np.float)
        length = 10
        states = src.simple_model.Model.generate_emission(t, length)
        assert states[0] == 0

    def test_zero_exit_probability_gives_only_zero(self):
        t = np.array([[1, 0], [0.5, 0.5]], dtype=np.float)
        length = 10
        states = src.simple_model.Model.generate_emission(t, length)
        assert all(s == 0 for s in states)

    def test_zero_reentry_gives_only_ones(self):
        t = np.array([[0, 1], [0, 1]], dtype=np.float)
        length = 10
        states = src.simple_model.Model.generate_emission(t, length)
        assert states[0] == 0 and all(s == 1 for s in states[1:])

    @pytest.mark.flaky(reruns=10)
    def test_uniform_yields_uniform_states(self):
        t = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float)
        length = 10001
        states = src.simple_model.Model.generate_emission(t, length)
        assert sum(states[1:]) == pytest.approx((length - 1) / 2, rel=0.01)


testdata_overlaps_emission = [
    ([0, 0, 0, 0, 0], [(1, 3)], 0),
    ([1, 0, 0, 0, 1], [(1, 3)], 0),
    ([1, 1, 0, 0, 1], [(1, 3)], 1),
    ([1, 1, 1, 0, 1], [(1, 3)], 1),
    ([1, 0, 0, 1, 1], [(1, 3)], 0),
    ([1, 0, 0, 1, 1, 1, 0, 1], [(1, 3), (4, 7)], 1),
    ([1, 0, 0, 1, 0, 0, 0, 1], [(1, 3), (4, 7)], 0),
    ([1, 0, 1, 1, 0, 1, 0, 1], [(1, 3), (4, 7)], 2),
]


@pytest.mark.parametrize("states,intervals,expected", testdata_overlaps_emission)
def test_count_overlaps_emission(states, intervals, expected):
    result = src.simple_model.Model.count_overlaps_emission(intervals, states)
    assert result == expected


class TestEvalProbsSingleSimulation:
    def test_sum_is_one(self):
        fun = src.simple_model.Model.eval_probs_single_simulation
        r = [(0, 1)]
        q = [(0, 1)]
        chr_size = 10
        result = fun(r, q, chr_size, tries=100)
        assert logsumexp(result) == pytest.approx(0)


class TestProbsSingleSimulationPerm:
    testdata = [
        ([(0, 1)], [(0, 1)], 1, [-np.inf, 0]),
        ([(0, 1)], [(0, 1)], 2, np.log([0.5, 0.5])),
        ([(0, 1)], [(0, 1)], 3, np.log([2 / 3, 1 / 3])),
        ([(0, 1)], [(0, 1)], 4, np.log([3 / 4, 1 / 4])),
        ([(0, 1)], [(0, 1)], 5, np.log([4 / 5, 1 / 5])),
        ([(0, 1), (2, 3)], [(0, 1)], 3, np.log([1 / 3, 2 / 3, 0])),
        ([(0, 1), (2, 3)], [(0, 1)], 4, np.log([1 / 2, 1 / 2, 0])),
        ([(0, 1), (2, 3)], [(0, 1), (2, 3)], 3, np.log([0, 0, 1])),
        ([(0, 1), (2, 3)], [(0, 1), (2, 3)], 4, np.log([1 / 3, 1 / 3, 1 / 3])),
        ([(0, 1), (2, 3)], [(0, 1), (2, 3), (4, 5)], 5, np.log([0, 0, 1])),
        ([(0, 1), (3, 4)], [(0, 1), (2, 3), (4, 5)], 5, np.log([0, 1, 0])),
        ([(0, 8)], [(0, 1)], 10, np.log([0.2, 0.8])),
    ]

    def test_sum_is_one(self):
        fun = src.simple_model.Model.eval_probs_single_simulation_perm
        r = [(0, 1)]
        q = [(0, 1), (2, 4)]
        chr_size = 10
        result = fun(r, q, chr_size, tries=100)
        assert logsumexp(result) == pytest.approx(0)

    @pytest.mark.parametrize("r,q,chr_size,expected", testdata)
    def test_correct_probs(self, r, q, chr_size, expected):
        fun = src.simple_model.Model.eval_probs_single_simulation_perm
        result = fun(r, q, chr_size, tries=1000)
        assert np.exp(result) == pytest.approx(np.exp(expected), abs=0.05)

    small_inputs = [
        ([(0, 1)], [(3, 5)], 5),
        ([(0, 1)], [(3, 5)], 10),
        ([(0, 1)], [(3, 5)], 20),
        ([(0, 1)], [(3, 5)], 30),
        ([(0, 1), (2, 3)], [(3, 6)], 6),
        ([(0, 1), (2, 3)], [(3, 6)], 10),
        ([(0, 1), (2, 3)], [(3, 6)], 20),
        ([(0, 1), (2, 3)], [(3, 6)], 30),
        ([(0, 1), (2, 3)], [(3, 6), (1, 2)], 6),
        ([(0, 1), (2, 3)], [(3, 6), (1, 2)], 12),
        ([(0, 1), (2, 3)], [(3, 6), (1, 2)], 20),
        ([(0, 1), (2, 3)], [(3, 6), (1, 2)], 30),
        ([(0, 1), (2, 3), (7, 10)], [(3, 6), (1, 2), (8, 10)], 10),
        ([(0, 1), (2, 3), (7, 10)], [(3, 6), (1, 2), (8, 10)], 20),
        ([(0, 1), (2, 3), (7, 10)], [(3, 6), (1, 2), (8, 10)], 30),
        ([(0, 1), (2, 3), (7, 10)], [(3, 6), (1, 2), (8, 10)], 50),
        ([(0, 1), (2, 3), (7, 10), (16, 22)], [(3, 6), (1, 2), (8, 10), (11, 20)], 22),
        ([(0, 1), (2, 3), (7, 10), (16, 22)], [(3, 6), (1, 2), (8, 10), (11, 20)], 32),
        ([(0, 1), (2, 3), (7, 10), (16, 22)], [(3, 6), (1, 2), (8, 10), (11, 20)], 42),
    ]

    @pytest.mark.parametrize("r,q,chr_size", small_inputs)
    @pytest.mark.flaky(reruns=5)
    def test_similar_to_exact_perm(self, r, q, chr_size):
        exact_res = simple_model.Model.eval_probs_single_exact_perm(r, q, chr_size)
        simulation_res = \
            simple_model.Model.eval_probs_single_simulation_perm(r, q, chr_size, tries=10000)
        assert np.exp(simulation_res) == pytest.approx(np.exp(exact_res), abs=0.05)


class TestEvalProbsSingleDirect:
    funs = {
        "simulation": src.simple_model.Model.eval_probs_single_simulation,
        "direct": src.simple_model.Model.eval_probs_single_direct_lm,
    }

    testdata = [
        ([(1, 2)], [(1, 2)], 3),
        ([(1, 2)], [(2, 3)], 3),
        ([(1, 2)], [(2, 3)], 10),
        ([(1, 2)], [(2, 3), (5, 9)], 10),
        ([(1, 2), (6, 7)], [(2, 3), (5, 9)], 20),
        ([(1, 2), (6, 7), (10, 11), (15, 16), (17, 18)], [(2, 3), (5, 9), (19, 20)], 20),
    ]

    @pytest.mark.parametrize("r,q,c", testdata)
    def test_sum_is_one(self, r, q, c):
        fun = self.funs['direct']
        chr_size = c
        result = fun(r, q, chr_size)
        assert logsumexp(result) == pytest.approx(0)

    @pytest.mark.parametrize("r,q,chr_size", testdata)
    @pytest.mark.flaky(reruns=5)
    def test_close_to_simulation(self, r, q, chr_size):
        result_sim = self.funs["simulation"](r, q, chr_size, tries=100)
        result_direct = self.funs["direct"](r, q, chr_size)
        assert np.exp(result_direct) == pytest.approx(np.exp(result_sim), abs=0.05)


class TestEvalProbsSingleDirectEigen:
    funs = {
        "simulation": src.simple_model.Model.eval_probs_single_simulation,
        "direct": src.simple_model.Model.eval_probs_single_direct_lm_eigen,
    }

    testdata = [
        ([(1, 2)], [(1, 2)], 3),
        ([(1, 2)], [(2, 3)], 3),
        ([(1, 2)], [(2, 3)], 10),
        ([(1, 2)], [(2, 3), (5, 9)], 10),
        ([(1, 2), (6, 7)], [(2, 3), (5, 9)], 20),
        ([(1, 2), (6, 7), (10, 11), (15, 16), (17, 18)], [(2, 3), (5, 9), (19, 20)], 20),
        ([(1, 2), (6, 7), (10, 11), (15, 16), (17, 18)], [(2, 3), (5, 9), (19, 20)], 200000),
    ]

    @pytest.mark.parametrize("r,q,c", testdata)
    def test_sum_is_one(self, r, q, c):
        fun = self.funs['direct']
        chr_size = c
        result = fun(r, q, chr_size)
        assert logsumexp(result) == pytest.approx(0)

    @pytest.mark.parametrize("r,q,chr_size", testdata)
    @pytest.mark.flaky(reruns=5)
    def test_close_to_simulation(self, r, q, chr_size):
        result_sim = self.funs["simulation"](r, q, chr_size, tries=100)
        result_direct = self.funs["direct"](r, q, chr_size)
        assert np.exp(result_direct) == pytest.approx(np.exp(result_sim), abs=0.05)


testdata_direct_exp = [
    (np.array([[1, 0], [0, 1]]), 10),
    (np.array([[1, 2], [3, 4]]), 0),
    (np.array([[1, 2], [3, 4]]), 1),
    (np.array([[1, 2], [3, 4]]), 5),
    (np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]]), 5),
]


@pytest.mark.parametrize("a,n", testdata_direct_exp)
def test_direct_exp(a, n):
    result = src.simple_model.direct_exp(a, n)
    expected = src.simple_model.direct_slow_exp(a, n)
    assert result == pytest.approx(expected)


class TestRandomPartition:
    def test_negative_k_raises_error(self):
        with pytest.raises(ValueError):
            result = list(simple_model.random_partition(10, -1))

    @pytest.mark.parametrize("n", range(20))
    def test_one_partition_is_n(self, n):
        expected = [n]
        result = list(simple_model.random_partition(n, 1))
        assert result == expected

    @pytest.mark.parametrize("k", list(range(1, 20)))
    def test_n_partitions_of_zero_is_zero(self, k):
        expected = [0 for _ in range(k)]
        result = list(simple_model.random_partition(0, k))
        assert result == expected

    @pytest.mark.parametrize("n,k", [(10, 1), (20, 2), (3, 3), (10, 5)])
    def test_mean_value_is_equal_for_each_position(self, n, k):
        tries = 10000
        expected_mean = n / k
        sums = np.zeros(k, dtype=int)
        for t in range(tries):
            result = list(simple_model.random_partition(n, k))
            for i in range(k):
                sums[i] += result[i]
        means = [s / tries for s in sums]
        expected = [expected_mean for _ in range(k)]
        assert means == pytest.approx(expected, rel=0.05)

    @pytest.mark.parametrize("n,k", [(1, 1), (2, 1), (1, 2), (2, 2),
                                     (10, 2), (1, 3), (2, 3), (3, 3)])
    def test_each_partition_has_the_same_probability(self, n, k):
        tries = 100000
        counts = {tuple(p): 0 for p in simple_model.generate_all_partitions(n, k)}
        for t in range(tries):
            p = simple_model.random_partition(n, k)
            counts[tuple(p)] += 1
        for p, c in counts.items():
            assert c / tries == pytest.approx(1 / len(counts), rel=0.1), f"{counts}"


class TestNumberOfPartitions:
    @pytest.mark.parametrize("n", [0, 1, 2, 5, 20, 100])
    def test_one_partition_is_always_one(self, n):
        result = simple_model.number_of_partitions(n, 1)
        assert result == 1

    @pytest.mark.parametrize("k", [1, 2, 5, 20, 100])
    def test_partition_of_zero_is_always_one(self, k):
        result = simple_model.number_of_partitions(0, k)
        assert result == 1

    @pytest.mark.parametrize("n,k", [(1, 2), (5, 2), (10, 3),
                                     (2, 16), (100, 100), (10000, 10000)])
    def test_recurrence_holds(self, n, k):
        whole = simple_model.number_of_partitions(n, k)
        from_reccurence = 0
        for i in range(n + 1):
            subresult = simple_model.number_of_partitions(n - i, k - 1)
            from_reccurence += subresult
        assert whole == pytest.approx(from_reccurence)

    @pytest.mark.skip(reason="It's too slow to check everytime")
    @pytest.mark.slow
    @pytest.mark.parametrize("n,k", [(10 ** 6, 1000), (10 ** 7, 10000)])
    def test_recurrence_holds_for_large_numbers(self, n, k):
        self.test_recurrence_holds(n, k)


class TestGenerateAllPartitions:
    @pytest.mark.parametrize("k", [0, -1, -2, -20, -100])
    def test_nonpositive_k_raises_value_error(self, k):
        with pytest.raises(ValueError):
            result = list(simple_model.generate_all_partitions(15, k))

    @pytest.mark.parametrize("k", [1, 2, 10, 20, 30])
    def test_partition_of_zero_is_a_zero_vector(self, k):
        expected = [[0 for _ in range(k)]]
        result = list(simple_model.generate_all_partitions(0, k))
        assert result == expected

    @pytest.mark.parametrize("k", [1, 2, 5, 10, 20])
    def test_partition_of_one_is_a_set_of_unit_vectors(self, k):
        expected = set(tuple(1 if j == i else 0 for j in range(k)) for i in range(k))
        result = set(tuple(p) for p in simple_model.generate_all_partitions(1, k))
        assert result == expected

    testdata = [
        (1, 1, [(1,)]),
        (2, 1, [(2,)]),
        (2, 2, [(0, 2), (1, 1), (2, 0), ]),
        (10, 2, [(i, 10 - i) for i in range(11)]),
        (2, 3, [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]),
    ]

    @pytest.mark.parametrize("n,k,expected", testdata)
    def test_small_cases(self, n, k, expected):
        result = set(tuple(p) for p in simple_model.generate_all_partitions(n, k))
        assert result == set(expected)

    testdata2 = [
        (5, 8),
        (10, 2),
        (20, 2),
        (20, 5),
        (10, 10),
    ]

    @pytest.mark.parametrize("n,k", testdata2)
    def test_number_of_partitions_is_correct(self, n, k):
        expected = simple_model.number_of_partitions(n, k)
        result = 0
        for item in simple_model.generate_all_partitions(n, k):
            result += 1
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("n,k", testdata2)
    def test_partitions_are_unique(self, n, k):
        all_partitions = list(simple_model.generate_all_partitions(n, k))
        expected = len(all_partitions)
        unique_count = len(set(map(tuple, all_partitions)))
        assert unique_count == expected


class TestEvalProbsSingleExactPerm:
    testdata = [
        ([(0, 1)], [(0, 1)], 1, [-np.inf, 0]),
        ([(0, 1)], [(0, 1)], 2, np.log([0.5, 0.5])),
        ([(0, 1)], [(0, 1)], 3, np.log([2 / 3, 1 / 3])),
        ([(0, 1)], [(0, 1)], 4, np.log([3 / 4, 1 / 4])),
        ([(0, 1)], [(0, 1)], 5, np.log([4 / 5, 1 / 5])),
        ([(0, 1), (2, 3)], [(0, 1)], 3, np.log([1 / 3, 2 / 3, 0])),
        ([(0, 1), (2, 3)], [(0, 1)], 4, np.log([1 / 2, 1 / 2, 0])),
        ([(0, 1), (2, 3)], [(0, 1), (2, 3)], 3, np.log([0, 0, 1])),
        ([(0, 1), (2, 3)], [(0, 1), (2, 3)], 4, np.log([1 / 3, 1 / 3, 1 / 3])),
        ([(0, 1), (2, 3)], [(0, 1), (2, 3), (4, 5)], 5, np.log([0, 0, 1])),
        ([(0, 1), (3, 4)], [(0, 1), (2, 3), (4, 5)], 5, np.log([0, 1, 0])),
        ([(0, 8)], [(0, 1)], 10, np.log([0.2, 0.8])),
    ]

    def test_sum_is_one(self):
        fun = src.simple_model.Model.eval_probs_single_exact_perm
        r = [(0, 1)]
        q = [(0, 1), (2, 4)]
        chr_size = 10
        result = fun(r, q, chr_size)
        assert logsumexp(result) == pytest.approx(0)

    @pytest.mark.parametrize("r,q,chr_size,expected", testdata)
    def test_correct_probs(self, r, q, chr_size, expected):
        fun = src.simple_model.Model.eval_probs_single_exact_perm
        result = fun(r, q, chr_size)
        assert np.exp(result) == pytest.approx(np.exp(expected))


class TestCdfFirstInPartition:
    @pytest.mark.parametrize("n,k", [(5, 8), (20, 5), (150, 220)])
    def test_cdf_matches_sum_of_probs(self, n, k):
        distro = simple_model.probs_first_in_partition(n, k)
        expected_cdf = list(itertools.accumulate(distro))
        cdf = [simple_model.cdf_first_in_partition(i, n, k) for i in range(n + 1)]
        assert cdf == pytest.approx(expected_cdf)


class TestFindArgInfimum:
    testdata = [
        (lambda x: x, 128, 0, 1),
        (lambda x: x, 10, 0, 20),
        (lambda x: x ** 2, 10, 0, 20),
        (lambda x: x ** 2 + 4.7, 10, 0, 20),
        (lambda i: [0.2, 0.7, 0.8][i], 0.65, 0, 3),
    ]

    @staticmethod
    def slow(f, x, lower, upper):
        res = lower
        while f(res) < x and res + 1 < upper:
            res += 1
        return res

    @pytest.mark.parametrize("f,x,lower,upper", testdata)
    def test_find_arg_infimum(self, f, x, lower, upper):
        expected = TestFindArgInfimum.slow(f, x, lower, upper)
        result = simple_model.find_arg_infimum(f, x, lower, upper)
        assert result == expected


class TestFindA1Distribution:
    @pytest.mark.parametrize("n,k", [(10, 2), (10, 3), (10, 4), (10, 5), (100, 70), (0, 16)])
    def test_is_a1_beta_binomial(self, n, k):
        a1_real = simple_model.probs_first_in_partition(n, k)
        alpha = 1
        beta = k - 1
        bb = [scipy.stats.betabinom.pmf(i, n, alpha, beta) for i in range(n + 1)]
        assert bb == pytest.approx(a1_real)


class TestTwoLineDP:
    def test_copying_is_safe(self):
        m = 4
        prev_line = np.array([[0, 0] for _ in range(m + 1)], dtype=np.longdouble)
        next_line = np.array([[0, 0] for _ in range(m + 1)], dtype=np.longdouble)

        prev_line[0] = np.array([1, 2])
        assert all(prev_line[0] == [1, 2])
        next_line = 2 * prev_line
        assert all(next_line[0] == [2, 4])
        prev_line = next_line
        next_line = 2 * prev_line
        assert all(next_line[0] == [4, 8])
        assert all(prev_line[0] == [2, 4])


class TestEigenMatrixExpTD:
    tests = [
        (0.5, 0.5, 0),
        (0.5, 0.5, 1),
        (0.5, 0.5, 2),
        (0.5, 0.5, 3),
        (0.5, 0.5, 4),
        (0.5, 0.5, 20),
        (0.5, 0.5, 100),
        (0.5, 0.5, 1000),
        (0.2, 0.8, 10),
        (0.2, 0.8, 100),
        (0.2, 0.8, 1000),
        (0.2, 1, 1000),
        (0, 1, 1000),
    ]

    speed_tests = [
        (0.5, 0.5, 5),
        (0.5, 0.5, 20),
        (0.5, 0.5, 100),
        (0.5, 0.5, 1000),
        (0.5, 0.5, 10000),
        (0.5, 0.5, 100000),
        (0.5, 0.5, 1000000),
        (0.5, 0.5, 10000000),
    ]

    @pytest.mark.parametrize("x,y,a", tests)
    def test_direct_exp_t_eigen_correct(self, x, y, a):
        T = np.array([[x, 1-x], [1-y, y]], dtype=np.longdouble)
        expected = src.simple_model.direct_exp(T, a)

        E = src.simple_model.TDExp(x, y)
        result = E.exp_t(a)

        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("x,y,a", tests)
    def test_direct_exp_d_eigen_correct(self, x, y, a):
        D = np.array([[x, 0], [1 - y, 0]], dtype=np.longdouble)
        expected = src.simple_model.direct_exp(D, a)

        E = src.simple_model.TDExp(x, y)
        result = E.exp_d(a)

        assert result == pytest.approx(expected)

    @pytest.mark.skip(reason="Only used to estimate the threshold of switching from log to diag")
    @pytest.mark.parametrize("x,y,a", speed_tests)
    def test_eigen_exp_t_is_faster(self, x, y, a):
        tries = 2000
        T = np.array([[x, 1-x], [1-y, y]], dtype=np.longdouble)
        t_start = datetime.datetime.now()
        for _ in range(tries):
            res = src.simple_model.direct_exp(T, a)
        t_end = datetime.datetime.now()
        old_avg_time = (t_end - t_start)/tries

        E = src.simple_model.TDExp(x, y)
        t_start = datetime.datetime.now()
        for _ in range(tries):
            res = E.exp_t(a)
        t_end = datetime.datetime.now()
        new_avg_time = (t_end - t_start) / tries

        assert new_avg_time < old_avg_time

    @pytest.mark.skip(reason="Only used to estimate the threshold of switching from log to diag")
    @pytest.mark.parametrize("x,y,a", speed_tests)
    def test_eigen_exp_d_is_faster(self, x, y, a):
        tries = 2000
        D = np.array([[x, 0], [1 - y, 0]], dtype=np.longdouble)
        t_start = datetime.datetime.now()
        for _ in range(tries):
            res = src.simple_model.direct_exp(D, a)
        t_end = datetime.datetime.now()
        old_avg_time = (t_end - t_start) / tries

        E = src.simple_model.TDExp(x, y)
        t_start = datetime.datetime.now()
        for _ in range(tries):
            res = E.exp_d(a)
        t_end = datetime.datetime.now()
        new_avg_time = (t_end - t_start) / tries

        assert new_avg_time < old_avg_time