# MCDP

## Installation

```shell
conda install mamba
mamba create -c conda-forge -c bioconda -n mcdp python=3.9 scipy pytest pytest-rerunfailures argh numpy matplotlib yaml pandas pybigwig xlrd
conda activate mcdp
```

## Demo

```shell
cd resources/examples_github
conda activate mcdp
bash run_comparison.sh
```

## Usage

```
usage: mcdp.py [-h] [-l LOG] ref_intervals query_intervals chr_sizes

MCDP: compute p-value of number of overlaps of two interval sets.

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
    

positional arguments:
  ref_intervals      List of reference intervals
  query_intervals    List of query intervals
  chr_sizes          List of chromosome sizes

optional arguments:
  -h, --help         show this help message and exit
  -l LOG, --log LOG  Log file (default: -)
```