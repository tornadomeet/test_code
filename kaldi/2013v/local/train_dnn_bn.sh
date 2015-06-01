#!/bin/bash

###
### First we need to dump the fMLLR features, so we can train on them easily
###
train_cmd=run.pl
decode_cmd=run.pl

featType=$1
proFeat=delta

langDir=$2
gmmdir=$3
ali=$4

if [ $featType = "mfcc" ];then
	proFeat=lda
fi


false && \
{
#gmmdir=exp/tri2b

# test
#dir=data-fmllr-tri3b-fbank/test
#steps/make_fmllr_opt_feats.sh --nj 10 \
#   $dir data_fbank/test $gmmdir $dir/_log $dir/_data || exit 1

# train
dir=data-fmllr-tri3b-$featType/train
steps/make_fmllr_opt_feats.sh --nj 10 \
   $dir data_$featType/train $gmmdir $dir/_log $dir/_data $proFeat || exit 1

# split the data : 90% train 10% cross-validation (held-out)
utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
}
###
### Now we can pre-train stack of RBMs
### (small database, smaller DNN)
###

false && \
{ # Pre-train the DBN
dir=exp/tri3b_pretrain-dbn_$featType
steps/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 10 data-fmllr-tri3b-$featType/train $dir || exit 1;
}

###
### Now we train the DNN optimizing cross-entropy.
###
### before training, we should copy the final.feature_transform file form other place

#false && \
{ # Train the MLP
dir=exp/tri3b_dnn_angel_$featType
#ali=exp/tri2b_ali
#feature_transform=exp/tri3b_pretrain-dbn_$featType/final.feature_transform
feature_transform=final.feature_transform
#dbn=exp/tri3b_pretrain-dbn_$featType/6.dbn
dbn=null  #because we don't use pre-training in bn

# steps/train_nnet.sh --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
steps/train_nnet_bn.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
 data-fmllr-tri3b-$featType/train_tr90 data-fmllr-tri3b-$featType/train_cv10 $langDir $ali $ali $dir || exit 1;
# decode (reuse HCLG graph)
#steps/decode_nnet.sh --nj 20 --config conf/decode_dnn.config --acwt 0.1 \
# exp/tri2b/graph data-fmllr-tri3b-fbank/test $dir/decode || exit 1;
}



###
### Finally we train using sMBR criterion.
### We do Stochastic-GD with per-utterance updates. 
### Use acwt 0.1, although it is not the best-WER value.
###

dir=exp/tri3b_pretrain-dbn_dnn_smbr_$featType
srcdir=exp/tri3b_pretrain-dbn_dnn_$featType
acwt=0.1

# First we need to generate lattices and alignments:
false && \
{
steps/align_nnet.sh --nj 20 \
  data-fmllr-tri3b-$featType/train $langDir $srcdir ${srcdir}_ali || exit 1;
steps/make_denlats_nnet.sh --nj 20 --config conf/decode_dnn.config --acwt $acwt \
  data-fmllr-tri3b-$featType/train $langDir $srcdir ${srcdir}_denlats  || exit 1;
}
# Now we re-train the DNN by 6 iterations of sMBR 
false && \
{
steps/train_nnet_mpe.sh --num-iters 1 --acwt $acwt --do-smbr true \
  data-fmllr-tri3b-$featType/train $langDir $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
}
# Decode
false && \
{
for ITER in 1 2 3 4 5 6; do
  # decode
  steps/decode_nnet.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri2b/graph data-fmllr-tri3b-fbank/test $dir/decode_it${ITER} || exit 1
done 
}


# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
