#!/bin/bash

. ./cmd.sh

feat_dir=/home/hzchenhaibo/sourcecode/kaldi-trunk/egs/rm/test_use/data_fbank/train
lang_dir=/home/hzchenhaibo/sourcecode/kaldi-trunk/egs/rm/test_use/GMM4DNNTest/lang/train
ali_dir=/home/hzchenhaibo/sourcecode/kaldi-trunk/egs/rm/test_use/GMM4DNNTest/exp/tri2b_ali

steps/nnet2/train_tanh_gpu.sh \
	--num-jobs-nnet 2 --num-threads 1 \
	--splice-width 5 \
	--initial-learning-rate 0.02 --final-learning-rate 0.004 \
	--num-hidden-layers 6 --hidden-layer-dim 1024 \
	--num-epochs-extra 10 --add-layers-period 1 \
	--mix-up 4000 \
	--cmd "$decode_cmd"
	$feat_dir $lang_dir $ali_dir exp/nnet4b_gpu  || exit 1


