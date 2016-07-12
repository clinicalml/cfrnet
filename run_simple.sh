#!/bin/sh

p_alpha=1e-2
p_lambda=1e-3
n_in=2
n_out=2
dropout_in=1.0
dropout_out=1.0
lrate=0.01
lrate_decay=0.92
decay=0.5
batch_size=100
dim_in=25
dim_out=25
rbf_sigma=0.1
imb_fun=wass
wass_lambda=1
wass_iterations=10
wass_bpt=1
use_p_correction=1
iterations=2000
weight_init=0.001
outbase='results'
datapath='data/ihdp_sample.csv'
loss=l2
sparse=0
varsel=0
save_rep=0

timestamp=$(date +'%Y%m%d_%H%M%S-%3N')

outdir="$outbase/single_${n_in}-${n_out}_${imb_fun}_VS${varsel}_$timestamp"

param="--n_in $n_in --n_out $n_out --p_lambda $p_lambda --p_alpha $p_alpha"
param="$param --dropout_in $dropout_in --dropout_out $dropout_out --lrate $lrate"
param="$param --decay $decay --batch_size $batch_size --dim_in $dim_in"
param="$param --dim_out $dim_out --imb_fun $imb_fun --rbf_sigma $rbf_sigma --wass_lambda $wass_lambda"
param="$param --iterations $iterations --weight_init $weight_init"
param="$param --datapath $datapath --outdir $outdir --wass_bpt $wass_bpt"
param="$param --lrate_decay $lrate_decay --loss $loss --wass_iterations $wass_iterations"
param="$param --use_p_correction $use_p_correction --varsel $varsel --save_rep $save_rep"

mkdir -p results
mkdir $outdir
python cfr_train_simple.py $param
