import itertools
import logging
import sys
import warnings
from collections import defaultdict

import numpy as np
import numpy.random
import pytest
import scipy.stats
from numpy import logaddexp
from scipy.special import logsumexp, loggamma

import helpers
from helpers import select_intervals_by_chr_name

logger = logging.getLogger("root")

class Model:
    def __init__(self, ref_intervals, query_intervals, chr_sizes, method="direct", tries=1000):
        self.ref_intervals = ref_intervals
        self.query_intervals = query_intervals
        self.chr_sizes = chr_sizes

        self.prob_methods = {
            "direct": self.eval_probs_single_direct_lm,
            "direct_eigen": self.eval_probs_single_direct_lm_eigen,
            "log": self.eval_probs_single_log,
            "simulation": lambda p, q, c:
                self.eval_probs_single_simulation(p, q, c, tries),
            "simulation_perm": lambda p, q, c:
                self.eval_probs_single_simulation_perm(p, q, c, tries),
            "exact_perm": self.eval_probs_single_exact_perm,
            "sim_perm_zero": lambda p, q, c:
                self.eval_probs_single_simulation_perm_zero(p, q, c, tries),
        }
        if method not in self.prob_methods:
            raise ValueError(f"Unknown p-value calculation method '{method}! "
                             f"Available: {self.prob_methods.keys()}'")
        self.prob_method = self.prob_methods[method]

    def eval_pvalue(self, overlap_count):
        # we may precompute some things later, but for now it doesn't really matter
        # calculate p-value for each chromosome
        probs_by_chromosome = []
        for chr_name, chr_size in self.chr_sizes:
            # the selection can be optimised, but we are in O(m^2) time complexity anyway
            r = select_intervals_by_chr_name(self.ref_intervals, chr_name)
            q = select_intervals_by_chr_name(self.query_intervals, chr_name)
            probs = self.prob_method(r, q, chr_size)
            probs_by_chromosome.append(probs)

        # calculate joint p-value
        joint_pvalue = helpers.joint_pvalue(probs_by_chromosome, overlap_count)
        return joint_pvalue

    def eval_sf(self):
        probs_by_chromosome = []
        for chr_name, chr_size in self.chr_sizes:
            logger.debug(f"Started computing sf for chromosome '{chr_name}'...")
            # the selection can be optimised (now it's O(Cm)),
            # but we are in O(m^2) time complexity anyway
            r = select_intervals_by_chr_name(self.ref_intervals, chr_name)
            q = select_intervals_by_chr_name(self.query_intervals, chr_name)
            logger.debug(f"m={len(r)}, n={len(q)}")
            probs = self.prob_method(r, q, chr_size)
            probs_by_chromosome.append(probs)
        logger.debug("Started computing joint logprobs...")
        joint_logprobs = helpers.joint_logprobs(probs_by_chromosome)
        logresult = np.zeros(len(joint_logprobs))
        logresult[-1] = joint_logprobs[-1]
        for i in reversed(range(len(joint_logprobs)-1)):
            logresult[i] = logaddexp(logresult[i+1], joint_logprobs[i])
        result = np.exp(logresult)
        return result

    @staticmethod
    def eval_probs_single_direct_lm(r, q, chr_size):
        if len(q) == 0 or len(r) == 0:
            return [0]

        T, D = get_direct_transition_matrices(chr_size, q)
        m = len(r)
        if r[0][0] == 0:
            warnings.warn("First reference interval starts with zero, changing to one!")
            r[0] = (1, r[0][1])
            if r[0][1] - r[0][0] == 0:
                warnings.warn("First reference interval has length 0, removing it!")
                r = r[1:]

        r_augmented = [(-np.inf, 0)] + r + [(chr_size, np.inf)]
        prev_line = np.array([[0, 0] for _ in range(m + 1)],
                     dtype=np.longdouble)
        prev_line[0, 0] = 1
        last_col = np.array([[0, 0] for _ in range(m+1)],
                            dtype=np.longdouble)

        # zero layer of DP P[*, 0, *] should be calculated in a separate way
        for j in range(1, m + 1):
            g = r_augmented[j][0] - r_augmented[j - 1][1]
            if j == 1:
                g -= 1
            assert g >= 0, f"Expected non-negative `g`, got {g=} instead!"
            l = r_augmented[j][1] - r_augmented[j][0]
            assert l >= 0
            prev_line[j] = prev_line[j-1].dot(direct_exp(T, g).dot(direct_exp(D, l)))
        last_col[0] = prev_line[-1].copy()

        logger = logging.getLogger("root")
        next_line = np.array([[0, 0] for _ in range(m + 1)], dtype=np.longdouble)
        for k in range(1, m + 1):
            next_line[k-1] = [0, 0]
            if k % 10 == 0:
                logger.debug(f"Processing {k}-th line out of {m} rows of DP table...")
            for j in range(k, m + 1):
                g = r_augmented[j][0] - r_augmented[j - 1][1]  # gap length
                if j == 1:
                    g -= 1
                assert g >= 0
                l = r_augmented[j][1] - r_augmented[j][0]  # interval length
                assert l >= 0
                # dont_hit = P[j-1, k] * T^g * D^l
                dont_hit = next_line[j-1].dot(direct_exp(T, g)).dot(direct_exp(D, l))
                # hit = P[j-1, k-1] * T^g * (T^l - D^l)
                hit = prev_line[j - 1].dot(direct_exp(T, g)).dot(direct_exp(T, l) - direct_exp(D, l))
               # P[j, k] = dont_hit + hit
                next_line[j] = dont_hit + hit
            last_col[k, 0] = next_line[-1, 0]
            last_col[k, 1] = next_line[-1, 1]
            for j in range(m+1):
                prev_line[j, 0] = next_line[j, 0]
                prev_line[j, 1] = next_line[j, 1]
        probs = [np.log(np.sum(last_col[k, :])) for k in range(m + 1)]
        return probs

    @staticmethod
    def eval_probs_single_direct_lm_eigen(r, q, chr_size):
        if len(q) == 0 or len(r) == 0:
            return [0]

        x, y = estimate_mc_weights_simple(chr_size, q)
        E = TDExp(x, y)

        m = len(r)
        if r[0][0] == 0:
            warnings.warn("First reference interval starts with zero, changing to one!")
            r[0] = (1, r[0][1])
            if r[0][1] - r[0][0] == 0:
                warnings.warn("First reference interval has length 0, removing it!")
                r = r[1:]

        r_augmented = [(-np.inf, 0)] + r + [(chr_size, np.inf)]
        prev_line = np.array([[0, 0] for _ in range(m + 1)],
                             dtype=np.longdouble)
        prev_line[0, 0] = 1
        last_col = np.array([[0, 0] for _ in range(m + 1)],
                            dtype=np.longdouble)

        # zero layer of DP P[*, 0, *] should be calculated in a separate way
        for j in range(1, m + 1):
            g = r_augmented[j][0] - r_augmented[j - 1][1]
            if j == 1:
                g -= 1
            assert g >= 0, f"Expected non-negative `g`, got {g=} instead!"
            l = r_augmented[j][1] - r_augmented[j][0]
            assert l >= 0
            prev_line[j] = prev_line[j - 1].dot(E.exp_t(g).dot(E.exp_d(l)))
        last_col[0] = prev_line[-1].copy()

        logger = logging.getLogger("root")
        next_line = np.array([[0, 0] for _ in range(m + 1)], dtype=np.longdouble)
        for k in range(1, m + 1):
            next_line[k - 1] = [0, 0]
            if k % 10 == 0:
                logger.debug(f"Processing {k}-th line out of {m} rows of DP table...")
            for j in range(k, m + 1):
                g = r_augmented[j][0] - r_augmented[j - 1][1]  # gap length
                if j == 1:
                    g -= 1
                assert g >= 0
                l = r_augmented[j][1] - r_augmented[j][0]  # interval length
                assert l >= 0
                # dont_hit = P[j-1, k] * T^g * D^l
                dont_hit = next_line[j - 1].dot(E.exp_t(g).dot(E.exp_d(l)))
                # hit = P[j-1, k-1] * T^g * (T^l - D^l)
                hit = prev_line[j - 1].dot(E.exp_t(g)).dot(
                    E.exp_t(l) - E.exp_d(l))
                # P[j, k] = dont_hit + hit
                next_line[j] = dont_hit + hit
            last_col[k, 0] = next_line[-1, 0]
            last_col[k, 1] = next_line[-1, 1]
            for j in range(m + 1):
                prev_line[j, 0] = next_line[j, 0]
                prev_line[j, 1] = next_line[j, 1]
        probs = [np.log(np.sum(last_col[k, :])) for k in range(m + 1)]
        return probs

    @staticmethod
    def eval_probs_single_log(r, q, chr_size):
        raise NotImplementedError()
        # if len(q) == 0:
        #     # there are no query intervals. In that case we declare that no overlaps are possible
        #     return [0]
        #
        # T, D = Model.get_log_transition_matrices(chr_size, q)
        #
        # # DP table initialisation
        # # @TODO reduce the memory complexity to only two lines plus the last column
        # m = len(r)
        # r_augmented = [(-np.inf, 0)] + r + [(chr_size, np.inf)]
        # P = np.array([[[-np.inf, -np.inf] for _ in range(m+1)] for _ in range(m+1)],
        #              dtype=np.longdouble)
        # P[0, 0, 0] = 0
        #
        # # zero layer of DP P[*, 0, *] should be calculated in a separate way
        # for j in range(1, m+1):
        #     g = r_augmented[j-1][1] - r_augmented[j][0]
        #     if j == 1:
        #         g -= 1
        #     l = r_augmented[j][1] - r_augmented[j][0]
        #     P[j, 0] = log_multiply(P[j-1, 0], log_multiply(log_exp(T, g), log_exp(D, l)))
        #
        # for k in range(1, m+1):
        #     for j in range(1, m+1):
        #         g = r_augmented[j - 1][1] - r_augmented[j][0]  # gap length
        #         if j == 1:
        #             g -= 1
        #         l = r_augmented[j][1] - r_augmented[j][0]  # interval length
        #         # dont_hit = P[j-1, k] * T^g * D^l
        #         dont_hit = log_multiply(P[j - 1, k],
        #                                 log_multiply(
        #                                     log_exp(T, g),
        #                                     log_exp(D, l)
        #                                 )
        #                                 )
        #         # hit = P[j-1, k-1] * T^g * (T^l - D^l)
        #         hit = log_multiply(P[j - 1, k-1],
        #                            log_multiply(log_exp(T, g), log_diff(
        #                                log_exp(T, l), log_exp(D, l))))
        #         # P[j, k] = dont_hit + hit
        #         P[j, k] = log_sum(dont_hit, hit)
        # return [np.logsumexp(P[m, k, :]) for k in range(m+1)]

    @staticmethod
    def eval_probs_single_simulation(r, q, chr_size, tries=1):
        if len(q) == 0 or len(r) == 0:
            return [0]

        T, _ = get_direct_transition_matrices(chr_size, q)
        m = len(r)

        overlap_histogram = [0 for _ in range(m+1)]
        for t in range(tries):
            states = Model.generate_emission(T, r[-1][1])
            overlap_count = Model.count_overlaps_emission(r, states)
            overlap_histogram[overlap_count] += 1

        probs = [np.log(c/tries) for c in overlap_histogram]
        return probs

    @staticmethod
    def eval_probs_single_simulation_perm(r, q, chr_size, tries=1):
        if len(q) == 0 or len(r) == 0:
            return [0]

        n = len(q)
        m = len(r)

        query_lengths = [e - b for b, e in q]
        free_space = chr_size - sum(query_lengths) - (n-1)

        overlap_histogram = [0 for _ in range(m+1)]
        random_query_set = []
        for t in range(tries):
            gaps = random_partition(free_space, n+1)
            random_permutation = numpy.random.permutation(n)
            # print(random_permutation)
            random_query_set.clear()
            for i in range(n):
                b = (random_query_set[-1][1] if i > 0 else 0) \
                    + next(gaps) \
                    + (1 if i > 0 else 0)
                e = b + query_lengths[random_permutation[i]]
                random_query_set.append((b, e))
            #print(random_query_set)
            overlap_count = helpers.count_overlaps_single_chromosome(r, random_query_set)
            overlap_histogram[overlap_count] += 1

        probs = [np.log(c/tries) for c in overlap_histogram]
        return probs

    @staticmethod
    def eval_probs_single_simulation_perm_zero(r, q, chr_size, tries=1):
        if len(q) == 0 or len(r) == 0:
            return [0]

        n = len(q)
        m = len(r)

        query_lengths = [e - b for b, e in q]
        free_space = chr_size - sum(query_lengths)

        overlap_histogram = [0 for _ in range(m + 1)]
        random_query_set = []
        for t in range(tries):
            gaps = random_partition(free_space, n + 1)
            random_permutation = numpy.random.permutation(n)
            # print(random_permutation)
            random_query_set.clear()
            for i in range(n):
                b = (random_query_set[-1][1] if i > 0 else 0) \
                    + next(gaps)
                e = b + query_lengths[random_permutation[i]]
                random_query_set.append((b, e))
            # print(random_query_set)
            overlap_count = helpers.count_overlaps_single_chromosome(r, random_query_set)
            overlap_histogram[overlap_count] += 1

        probs = [np.log(c / tries) for c in overlap_histogram]
        return probs

    @staticmethod
    def eval_probs_single_exact_perm(r, q, chr_size):
        if len(q) == 0 or len(r) == 0:
            return [0]

        n = len(q)
        m = len(r)

        query_lengths = [e - b for b, e in q]
        free_space = chr_size - sum(query_lengths) - (n - 1)

        overlap_histogram = [0 for _ in range(m + 1)]
        random_query_set = []
        for gaps in generate_all_partitions(free_space, n+1):
            for permutation in itertools.permutations(range(n)):
                random_query_set.clear()
                for i in range(n):
                    b = (random_query_set[-1][1] if i > 0 else 0) \
                        + gaps[i] \
                        + (1 if i > 0 else 0)
                    e = b + query_lengths[permutation[i]]
                    random_query_set.append((b, e))
                # print(random_query_set)
                overlap_count = helpers.count_overlaps_single_chromosome(r, random_query_set)
                overlap_histogram[overlap_count] += 1
        total_count = sum(overlap_histogram)
        probs = [np.log(c / total_count) for c in overlap_histogram]
        return probs

    @staticmethod
    def generate_emission(t, chr_size):
        states = [0 for _ in range(chr_size)]
        for i in range(1, chr_size):
            prev_step = states[i - 1]
            next_step = scipy.stats.bernoulli.rvs(t[prev_step, 1])
            states[i] = next_step
        return states

    @staticmethod
    def get_intervals_from_emissions(emissions):
        last_b = -1
        is_open = False
        result = []
        for p, s in enumerate(itertools.chain(emissions, [0])):
            if s == 1 and not is_open:
                is_open = True
                last_b = p
            elif s == 0 and is_open:
                result.append((last_b, p))
                is_open = False
        return result

    @staticmethod
    def count_overlaps_emission(r, states):
        overlaps = 0
        for b, e in r:
            if sum(states[b:e]) > 0:
                overlaps += 1
        return overlaps


class TDExp:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.T = np.array([[x, 1 - x], [1 - y, y]], dtype=np.longdouble)
        self.D = np.array([[x, 0], [1 - y, 0]], dtype=np.longdouble)
        try:
            self.Q = np.array([[1, (x - 1) / (1 - y)], [1, 1]], dtype=np.longdouble)
            quotient = -2 + x + y
            self.Q_inv = np.array([[(-1 + y) / quotient, (-1 + x) / quotient],
                              [(1 - y) / quotient, (-1 + y) / quotient]], dtype=np.longdouble)
        except ZeroDivisionError as e:
            self.exp_t = lambda a: direct_exp(self.T, a)

        self.logx = np.log(x)

    def exp_t(self, a):
        if a < 0:
            raise ValueError(f"Exponent must be nonnegative, got '{a}' instead!")
        elif a == 0:
            return np.identity(2, dtype=np.longdouble)
        elif a == 1:
            return self.T.copy()
        # elif self.y == 1 or self.x == 1 or self.x + self.y == 2:
        #     return direct_exp(self.T, a)
        elif a < 100:
            return direct_exp(self.T, a)
        else:
            l = np.array([[1, 0], [0, np.power(self.x + self.y - 1, a)]], dtype=np.longdouble)
            result = self.Q.dot(l.dot(self.Q_inv))
            return result

    def exp_d(self, a):
        if a < 0:
            raise ValueError(f"Exponent must be nonnegative, got '{a}' instead!")
        elif a == 0:
            return np.identity(2, dtype=np.longdouble)
        elif a == 1:
            return self.D.copy()
        elif a < 50:
            return direct_exp(self.D, a)
        else:
            result = np.array([[np.exp(a * self.logx), 0],
                               [(1 - self.y) * np.exp((a - 1) * self.logx), 0]], dtype=np.longdouble)
            return result


def number_of_partitions(n, k):
    return scipy.special.binom(n+k-1, k-1)


def random_partition(n, k):
    yield from random_partition_sequential(n, k)


def sample_first_in_partition_fast(n, k):
    x = numpy.random.rand()
    f = lambda i: cdf_first_in_partition(i, n, k)
    result = find_arg_infimum(f, x, 0, n+1)
    return result


def sample_first_in_partition_beta_binomial(n, k):
    #result = scipy.stats.betabinom.rvs(n=n, a=1, b=k-1)
    u = numpy.random.rand()
    x = 1 - np.power(u, 1/(k-1))
    # x = scipy.stats.beta.rvs(a=1, b=k-1)
    result = scipy.stats.binom.rvs(n=n, p=x)
    return result


def random_partition_sequential(n, k, sample_first=sample_first_in_partition_fast):
    if k <= 0:
        raise ValueError(f"k should be positive, got {k} instead!")
    remainder = n
    for j in range(k - 1):
        subk = k - j
        elem = sample_first(remainder, subk)
        yield elem
        remainder -= elem
    yield remainder


def sample_first_in_partition(n, k):
    probs = probs_first_in_partition(n, k)
    elem = numpy.random.choice(n + 1, p=probs)
    return elem


def cdf_first_in_partition(i, n, k):
    return 1 - np.exp(loggamma(n-i+k-1) + loggamma(n+1) - loggamma(n-i) - loggamma(n+k))


def find_arg_infimum(f, x, lower, upper):
    """Assume that function f is monotonically increasing."""
    if upper - lower == 1:
        #print(f"find {x=} {lower=} {upper=} {f(lower)=}")
        return lower
    middle = (upper - lower)//2 + lower
    m_value = f(middle)
    #print(f"find {x=} {lower=} {upper=} {middle=} {m_value=}")
    if x > m_value:
        return find_arg_infimum(f, x, middle+1, upper)
    elif x == m_value:
        return middle  # or maybe just return middle
    else:
        l_value = f(lower)
        if l_value < x:
            return find_arg_infimum(f, x, lower+1, middle+1)
        else:
            return lower


def probs_first_in_partition(n, k):
    probs = [np.exp(np.log(k-1) + loggamma(n+k-i-1) + loggamma(n+1) - loggamma(n+k) - loggamma(n-i+1))
             for i in range(n+1)]
    return probs


def random_partition_permutation(n, k):
    if k <= 0:
        raise ValueError(f"k should be positive, got {k} instead!")
    objects = [1 for _ in range(n)] + [0 for _ in range(k-1)]
    permuted = numpy.random.permutation(objects)
    last_opening = None
    for p, x in enumerate(itertools.chain(permuted, [0])):
        if last_opening is None and x == 1:
            last_opening = p
        elif x == 0:
            if last_opening is None:
                part_size = 0
            else:
                part_size = p - last_opening
            yield part_size
            last_opening = None


def generate_all_partitions(n, k):
    if k <= 0:
        raise ValueError(f"k should be positive, got {k} instead!")
    if k == 1:
        yield [n]
        return

    for i in range(n+1):
        for tail in generate_all_partitions(n-i, k-1):
            yield [i] + tail


def get_log_transition_matrices(chr_size, q):
    x, y = estimate_mc_weights_simple(chr_size, q)
    T = np.array([[np.log(x), np.log(1 - x)], [np.log(1 - y), np.log(y)]],
                 dtype=np.longdouble)
    D = np.array([[np.log(x), -np.inf], [np.log(1 - y), -np.inf]],
                 dtype=np.longdouble)
    return T, D


def get_direct_transition_matrices(chr_size, q):
    x, y = estimate_mc_weights_simple(chr_size, q)
    T = np.array([[x, 1-x], [1-y, y]], dtype=np.longdouble)
    D = np.array([[x, 0], [1-y, 0]], dtype=np.longdouble)
    return T, D


def estimate_mc_weights_simple(chr_size, q):
    if len(q) == 0:
        raise ValueError(f"Query interval set should be non-empty!")
    total_intervals_length = sum(e - b for b, e in q)
    total_gaps_length = chr_size - total_intervals_length
    n = len(q)
    alpha = - n - 1 + total_gaps_length
    beta = -n + total_intervals_length
    x = alpha / (alpha + n)
    y = beta / (beta + n)
    return x, y


def direct_slow_exp(a, n):
    if n < 0:
        raise ValueError(f"Power should be non-negative, got {n=}!")
    if n == 0:
        return np.identity(a.shape[0], dtype=np.longdouble)
    if n == 1:
        return a
    return a.dot(direct_slow_exp(a, n-1))


def direct_exp(a, n):
    if n < 0:
        raise ValueError(f"Power should be non-negative, got {n=}!")
    if n == 0:
        return np.identity(a.shape[0], dtype=np.longdouble)
    if n == 1:
         return a
    if n % 2 == 0:
        return direct_exp(a.dot(a), n//2)
    else:
        return a.dot(direct_exp(a.dot(a), n//2))


class DirectPermCounting:
    def __init__(self, ref_intervals, query_intervals, chr_sizes, tries=100):
        self.ref_intervals = ref_intervals
        self.query_intervals = query_intervals
        self.chr_sizes = chr_sizes
        self.tries = tries

    def eval_pvalue(self, overlap_count):
        sf = self.eval_sf()
        return sf[overlap_count] if overlap_count < len(sf) else 0

    def eval_sf(self):
        samples_by_chromosome = []
        for chr_name, chr_size in self.chr_sizes:
            logger.debug(f"Started computing sf for chromosome '{chr_name}'...")
            r = select_intervals_by_chr_name(self.ref_intervals, chr_name)
            q = select_intervals_by_chr_name(self.query_intervals, chr_name)
            logger.debug(f"m={len(r)}, n={len(q)}")
            probs = self.sample_overlap_counts(r, q, chr_size, self.tries)
            samples_by_chromosome.append(probs)
        logger.debug("Started computing joint logprobs...")

        histogram_total = defaultdict(int)
        for t in range(self.tries):
            total_sample = sum(ss[t] for ss in samples_by_chromosome)
            histogram_total[total_sample] += 1

        joint_logprobs = np.log([histogram_total[i] / self.tries for i in range(max(histogram_total.keys())+1)])

        logresult = np.zeros(len(joint_logprobs))
        logresult[-1] = joint_logprobs[-1]
        for i in reversed(range(len(joint_logprobs) - 1)):
            logresult[i] = logaddexp(logresult[i + 1], joint_logprobs[i])
        result = np.exp(logresult)
        return result

    @staticmethod
    def sample_overlap_counts(r, q, chr_size, tries=1):
        if len(q) == 0 or len(r) == 0:
            return [0 for _ in range(tries)]

        n = len(q)

        query_lengths = [e - b for b, e in q]
        free_space = chr_size - sum(query_lengths) - (n - 1)

        random_query_set = []
        result = []
        for t in range(tries):
            gaps = random_partition(free_space, n + 1)
            random_permutation = numpy.random.permutation(n)
            # print(random_permutation)
            random_query_set.clear()
            for i in range(n):
                b = (random_query_set[-1][1] if i > 0 else 0) \
                    + next(gaps) \
                    + (1 if i > 0 else 0)
                e = b + query_lengths[random_permutation[i]]
                random_query_set.append((b, e))
            # print(random_query_set)
            overlap_count = helpers.count_overlaps_single_chromosome(r, random_query_set)
            result.append(overlap_count)

        return result
