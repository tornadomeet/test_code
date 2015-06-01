#!/bin/bash

nj=20
for i in `seq $nj`;do
   find exp/tri3b_pretrain-dbn_dnn_fbank_denlats//lat${i} -name *.gz | awk -v FS="/" '{ print gensub(".gz","","",$NF)" gunzip -c "$0" |"; }'
done >exp/tri3b_pretrain-dbn_dnn_fbank_denlats/lat.scp

