import logging
import logging.handlers

import argh

import simple_model
from helpers import load_intervals, load_chr_sizes, count_overlaps, merge_nondisjoint_intervals, \
    filter_intervals_by_chr_name, filter_empty_intervals


@argh.arg("ref_intervals", help="List of reference intervals")
@argh.arg("query_intervals", help="List of query intervals")
@argh.arg("chr_sizes", help="List of chromosome sizes")
@argh.arg("-l", "--log", help="Log file")
@argh.arg("-c", "--closed", help="If set, the intervals are treated as 1-based and closed")
@argh.arg("-s", "--sf", help="If set, the survival function would be dumped into the file")
@argh.arg("-m", "--method", choices=["direct_eigen", "sim_perm_nc"],
          help="Method for computing the p-value")
@argh.arg("-t", "--tries", help="Number of trials for simulation methods (for each chromosome)")
def main(ref_intervals,
         query_intervals,
         chr_sizes,
         log=None,
         closed=False,
         sf=None,
         method="direct_eigen",
         tries=100):
    """MCDP: compute p-value of number of overlaps of two interval sets.

Both reference and query interval files should be tab-separated files with three columns:

    1. Chromosome name (same as in `chr_sizes` file)
    2. Begin of an interval (0-based, closed)
    3. End of an interval (0-based, open)

Intervals should be non-overlapping and disjoint (i.e. there should be a positive gap between them).
If there is no gap (or intervals are overlapping), the program will merge those intervals.

Chromosome sizes list should be tab-separated with two columns:

    1. Chromosome name
    2. Length of a chromosome

Files can contain empty lines.
    """
    set_root_logger(log)
    logger = logging.getLogger("root")

    with open(ref_intervals) as f:
        logger.info(f"Loading reference interval set from '{ref_intervals}'...")
        ref_intervals = load_intervals(f, closed)
    with open(query_intervals) as f:
        logger.info(f"Loading query interval set from '{query_intervals}'...")
        query_intervals = load_intervals(f, closed)
    with open(chr_sizes) as f:
        logger.info(f"Loading chromosome sizes from '{chr_sizes}'...")
        chr_sizes = load_chr_sizes(f)

    raw_ref_count = len(ref_intervals)
    raw_query_count = len(query_intervals)

    chr_names = [chr_name for chr_name, chr_len in chr_sizes]
    ref_intervals = filter_intervals_by_chr_name(ref_intervals, chr_names)
    query_intervals = filter_intervals_by_chr_name(query_intervals, chr_names)

    ref_intervals = merge_nondisjoint_intervals(ref_intervals)
    query_intervals = merge_nondisjoint_intervals(query_intervals)

    ref_intervals = filter_empty_intervals(ref_intervals)
    query_intervals = filter_empty_intervals(query_intervals)

    logger.info(f"Number of reference intervals: {len(ref_intervals)} ({raw_ref_count} before merging)")
    logger.info(f"Number of query intervals: {len(query_intervals)} ({raw_query_count} before merging)")
    logger.info(f"Number of chromosomes: {len(chr_sizes)}")

    overlap_count = count_overlaps(ref_intervals, query_intervals)
    logger.info(f"Overlap count: {overlap_count}")

    if method == "sim_perm_nc":
        model = simple_model.DirectPermCounting(ref_intervals, query_intervals, chr_sizes, tries)
    else:
        model = simple_model.Model(ref_intervals, query_intervals, chr_sizes, method, tries)

    if sf is None:
        pvalue = model.eval_pvalue(overlap_count)
        logger.info(f"p-value: {pvalue}")
    else:
        sf_values = model.eval_sf()
        pvalue = sf_values[overlap_count]
        logger.info(f"p-value: {pvalue}")
        with open(sf, "w") as f:
            dump_sf(sf_values, f)


def set_root_logger(log_filename):
    logger = logging.getLogger("root")
    logger.setLevel(logging.DEBUG)
    add_console_handler(logger)

    if log_filename is not None:
        add_file_handler(log_filename, logger, level=logging.DEBUG)


def add_file_handler(log_filename, logger, level=logging.INFO):
    handler = logging.handlers.WatchedFileHandler(log_filename)
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def add_console_handler(logger, level=logging.DEBUG):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def dump_sf(sf_values, f):
    print("k\tpvalue", file=f)
    for k, value in enumerate(sf_values):
        print(f"{k}\t{value}", file=f)


if __name__ == "__main__":
    argh.dispatch_command(main)
