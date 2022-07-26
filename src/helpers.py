import itertools
import math
import warnings

import numpy as np
import scipy.signal
from scipy.special import logsumexp


def load_intervals(file, is_closed=False):
    result = []
    for lnum, line in enumerate(file):
        if len(line.strip()) == 0:
            # skip empty lines
            continue
        elements = line.strip().split("\t")
        if len(elements) != 3:
            raise ValueError(f"Incorrect number of columns! Line #{lnum}: {line}")
        chr_name, b, e = elements
        b = int(b)
        e = int(e)

        if is_closed:
            b -= 1

        if b < 0 or e < 0:
            raise ValueError(f"Coordinates should be non-negative! Line #{lnum}: {line}")
        if not (b < e):
            raise ValueError(f"Begin should be less that end! Line #{lnum}: {line}")
        result.append((chr_name, b, e))
    return sorted(result)


def load_chr_sizes(file):
    result = []
    for lnum, line in enumerate(file):
        if len(line.strip()) == 0:
            # skip empty lines
            continue
        elements = line.strip().split("\t")
        if len(elements) != 2:
            raise ValueError(f"Incorrect number of columns! Line #{lnum}: {line}")
        chr_name, length = elements
        length = int(length)
        if length <= 0:
            raise ValueError(f"Length should be positive! Line #{lnum}: {line}")
        result.append((chr_name, length))
    return result


def count_overlaps(r, q):
    chr_names = sorted(set(s[0] for s in itertools.chain(r, q)))
    r_sorted = sorted(r)
    q_sorted = sorted(q)
    r_next, q_next = 0, 0

    total_overlap_count = 0
    for chr_name in chr_names:
        while r_next < len(r_sorted) and r_sorted[r_next][0] < chr_name:
            r_next += 1
        while q_next < len(q_sorted) and q_sorted[q_next][0] < chr_name:
            q_next += 1
        r_sub = []
        while r_next < len(r_sorted) and r_sorted[r_next][0] == chr_name:
            r_sub.append(r_sorted[r_next][1:])
            r_next += 1
        q_sub = []
        while q_next < len(q_sorted) and q_sorted[q_next][0] == chr_name:
            q_sub.append(q_sorted[q_next][1:])
            q_next += 1
        if len(r_sub) == 0 or len(q_sub) == 0:
            continue
        overlap_count = count_overlaps_single_chromosome(r_sub, q_sub)
        total_overlap_count += overlap_count

    return total_overlap_count


def count_overlaps_single_chromosome(r, q):
    """Assumes that the input arrays are sorted."""
    ends = []
    for b, e in r:
        ends.append((b, 0, 0))
        ends.append((e, 0, 1))
    for b, e in q:
        ends.append((b, 1, 0))
        ends.append((e, 1, 1))
    ends.append((math.inf, 1, 0))  # to count down the last possible overlap

    count = 0
    is_ref_interval_open = False
    is_query_interval_open = False
    is_current_ref_interval_counted = False
    last_pos = -1
    for pos, t, e in sorted(ends):
        if last_pos < pos:
            last_pos = pos
            if is_ref_interval_open \
                    and is_query_interval_open \
                    and not is_current_ref_interval_counted:
                count += 1
                is_current_ref_interval_counted = True
        # a new reference interval starts
        if t == 0 and e == 0:
            is_current_ref_interval_counted = False
            is_ref_interval_open = True
        # reference interval ends
        if t == 0 and e == 1:
            is_ref_interval_open = False
        # query interval starts
        if t == 1 and e == 0:
            is_query_interval_open = True
        # query interval ends
        if t == 1 and e == 1:
            is_query_interval_open = False
    return count


def joint_pvalue(probs_by_level, overlap_count):
    """Calculates joint p-value for a given `overlap_count`.
`pvalues_by_level` should contain log-values."""
    if overlap_count < 0:
        return 1
    logprobs = joint_logprobs(probs_by_level)
    if overlap_count >= len(logprobs):
        return 0
    result = np.exp(logsumexp(logprobs[overlap_count:]))
    return result


def joint_logprobs(probs_by_level):
    if len(probs_by_level) == 0:
        raise ValueError(f"p-values should have at least one level!")

    # check for empty levels
    if any(len(level) == 0 for level in probs_by_level):
        raise ValueError(f"Layers should be non-empty!")

    # if only one level, no need to do complex computations
    if len(probs_by_level) == 1:
        return probs_by_level[0]

    # now we have to do full DP
    # we want to preserve the memory, so instead of the whole DP table
    # we only store two levels
    max_k = sum(len(level) - 1 for level in probs_by_level)

    prev_row = np.array([-np.inf for _ in range(max_k+1)], dtype=np.longdouble)
    for pos, value in enumerate(probs_by_level[0]):
        prev_row[pos] = value

    current_row = np.array([-np.inf for _ in range(max_k+1)], dtype=np.longdouble)
    accum = []
    for level in probs_by_level[1:]:
        for k in range(max_k+1):
            for j in range(min(k+1, len(level))):
                accum.append(level[j] + prev_row[k-j])
            current_row[k] = logsumexp(accum)
            accum.clear()
        prev_row = current_row.copy()  # sic! without a copy it would just pass the reference
    return current_row


def select_intervals_by_chr_name(intervals, chr_name):
    result = []
    for name, b, e in intervals:
        if name == chr_name:
            result.append((b, e))
    return result


def merge_nondisjoint_intervals(intervals):
    intervals = filter(lambda interval: interval[1] < interval[2], intervals)
    intervals = sorted(intervals)
    if len(intervals) < 2:
        return intervals
    result = []
    c, b, e = intervals[0]
    for c2, b2, e2 in intervals[1:]:
        if c2 != c:
            result.append((c, b, e))
            c, b, e = c2, b2, e2
            continue
        if e < b2:
            result.append((c, b, e))
            c, b, e = c2, b2, e2
            continue
        e = max(e, e2)
    result.append((c, b, e))
    return result


def filter_intervals_by_chr_name(intervals, chr_names):
    result = [(c, b, e) for c, b, e in intervals if c in chr_names]
    return result


def filter_empty_intervals(intervals):
    results = [(c, b, e) for c, b, e in intervals if b < e]
    return results
