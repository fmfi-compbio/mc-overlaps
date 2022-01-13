#!/bin/bash

python ../../src/mcdp.py tcga_intervals.txt hirt_intervals.txt chr_sizes.txt --log hirt.log --sf pvalues_direct.txt
