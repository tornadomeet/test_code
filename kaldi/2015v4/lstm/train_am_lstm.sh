#/bin/bash

devData=/styx/home/hzwuw2014/kaldi-code/egs/rm/yixin/20150423_8k_lstm/gmm_train/data/test
trainData=/styx/home/hzwuw2014/kaldi-code/egs/rm/yixin/20150423_8k_lstm/gmm_train/data/train
langDir=/styx/home/hzwuw2014/kaldi-code/egs/rm/yixin/20150423_8k_lstm/gmm_train/lang/train

gmmDir=/styx/home/hzwuw2014/kaldi-code/egs/rm/yixin/20150423_8k_lstm/gmm_train/exp/tri2b

false && \
{
   echo "run_lstm.sh"
   local/nnet/run_lstm.sh $devData $trainData $langDir $gmmDir
}

featDir=data-fbank/train
srcDir=exp/lstm4f
dir=exp/lstm4smbr
acwt=0.1

false && \
{
   echo "nnet/align.sh"
   steps/nnet/align.sh --nj 20 --beam 40 --retry-beam 60 \
      $featDir $langDir $srcDir ${srcDir}_ali || exit 1;
}

false && \
{
   #echo "steps/make_denlats"
   #steps/make_denlats.sh --nj 25 --config conf/decode_dnn.config --acwt $acwt \
        #$featDir $langDir $srcDir ${srcDir}_denlats  || exit 1;

   echo "steps/make_denlats"
   steps/nnet/make_denlats_parallel.sh --nj 20 --sub-split 15 --num-threads 1 --config conf/decode_dnn.config --acwt $acwt \
        $featDir $langDir $srcDir ${srcDir}_denlats  || exit 1;
}

#false && \
{
   echo "steps/nnet/train_mpe.sh"
   steps/nnet/train_mpe.sh --num-iters 6 --learn-rate 0.000001 --acwt $acwt --do-smbr true \
      $featDir $langDir $srcDir ${srcDir}_ali ${srcDir}_denlats $dir || exit 1
}
