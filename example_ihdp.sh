#
# Requires download and extraction of IHDP_100. See README.md
#

mkdir results
mkdir results/example_ihdp

python cfr_param_search.py configs/example_ihdp.txt 20

python evaluate.py configs/example_ihdp.txt 1
