#!/bin/bash

# Copyright 2012  Karel Vesely (Brno University of Technology)
# Apache 2.0

# Train neural network

# Begin configuration.

# training options
learn_rate=0.008 # minibatch=128/GPU
momentum=0.0
l1_penalty=0
l2_penalty=0
# data processing
minibatch_size=256
randomizer_size=32768
# nnet_outdim=1024

#randomizer_size=2097152

#randomizer_size=4194304

#randomizer_size=262144

#randomizer_size=1048576
randomizer_seed=777
feature_transform=
# learn rate scheduling
max_iters=20
min_iters=
#start_halving_inc=0.5
#end_halving_inc=0.1
start_halving_impr=0.01
end_halving_impr=0.001
halving_factor=0.5
# misc.
verbose=1
# tool
#train_tool="nnet-train-frmshuff"
train_tool="/styx/home/hzwuw2014/Angel-0.2/bin/nnet-train-bn"
#train_tool="/styx/home/hzwuw2014/Angel-0.15/bin/nnet-train-bn"
 
# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

echo "number input params is:" $#
if [ $# != 6 ]; then
   echo "Usage: $0 <mlp-init> <feats-tr> <feats-cv> <labels-tr> <labels-cv> <exp-dir>"
   echo " e.g.: $0 0.nnet scp:train.scp scp:cv.scp ark:labels_tr.ark ark:labels_cv.ark exp/dnn1"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

mlp_init=$1
feats_tr=$2
feats_cv=$3
labels_tr=$4
labels_cv=$5
dir=$6
# echo "mlp_init=$mlp_init"

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

# Skip training
[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0

##############################
#start training

# choose mlp to start with
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}
echo "mlp_best model is : $mlp_best"
echo "mlp_base model is : $mlp_base"
# optionally resume training from the best epoch
[ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)

# print cv data
file_cv=`echo $feats_cv | awk '{print $2}' | cut -d':' -f2`
num_sentence_cv=`wc -l $file_cv | awk '{print $1}'`
echo "feats_cv file is: $feats_cv"
echo "labels_cv file is: $labels_cv"
echo "the number sentences of feats_cv is : $num_sentence_cv"

# print train data
file_tr=`echo $feats_tr | awk '{print $2}' | cut -d':' -f2`
num_sentence_tr=`wc -l $file_tr | awk '{print $1}'`
echo "feats_tr file is : $feats_tr"
echo "labels_tr file is : $labels_tr"
echo "the number sentences of feats_tr is : $num_sentence_tr"

# nnet_outdim=$(feat-to-dim "$file_cv nnet-forward $mlp_best scp:- scp:- |" - )  # failed
# echo "the output of train_nnet_scheduler.sh is:$nnet_outdim"

learn_rate=0.002 # 
# cross-validation on original network
$train_tool --cross-validate=true \
 --learn-rate=$learn_rate --momentum=$momentum \
 --minibatch-size=$(awk "BEGIN{print($minibatch_size*8)}") --randomizer-size=$(awk "BEGIN{print($randomizer_size*2)}") \
 --randomize=false --verbose=$verbose \
 --binary=false --num-sentence=$num_sentence_cv \
 --iter=0 \
 ${feature_transform:+ --feature-transform=$feature_transform} \
 "$feats_cv" "$labels_cv" "$mlp_best" \
 &> $dir/log/iter00.initial.log 

loss=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
loss_type=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')
echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $loss) $loss_type"


#false && \
{
# resume lr-halving
halving=0
[ -e $dir/.halving ] && halving=$(cat $dir/.halving)
# training
for iter in $(seq -w $max_iters); do
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  #echo "mlp_best=$mlp_best"
  #echo "mlp_next=$mlp_next"
#  echo "momentum=$momentum"

  # skip iteration if already done
  [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue 
  
#   --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
#   ${randomizer_seed:+ --randomizer-seed=$randomizer_seed} \
  # training
  $train_tool --cross-validate=false \
   --learn-rate=$learn_rate --momentum=$momentum \
   --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true --verbose=$verbose \
   --binary=false --num-sentence=$num_sentence_tr \
   --iter=$iter \
   ${feature_transform:+ --feature-transform=$feature_transform} \
   "$feats_tr" "$labels_tr" $mlp_best $mlp_next \
   &> $dir/log/iter${iter}.tr.log || exit 1; 
# done


 tr_loss=$(cat $dir/log/iter${iter}.tr.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
 echo -n "TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), "

# cross-validation
$train_tool --cross-validate=true \
 --learn-rate=$learn_rate --momentum=$momentum \
 --minibatch-size=$(awk "BEGIN{print($minibatch_size*8)}") --randomizer-size=$(awk "BEGIN{print($randomizer_size*2)}") \
 --randomize=false --verbose=$verbose \
 --binary=false --num-sentence=$num_sentence_cv \
 --iter=$iter \
 ${feature_transform:+ --feature-transform=$feature_transform} \
 "$feats_cv" "$labels_cv" "$mlp_next" \
 &> $dir/log/iter${iter}.cv.log || exit 1

  
 loss_new=$(cat $dir/log/iter${iter}.cv.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
 echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "

  # accept or reject new parameters (based on objective function)
  loss_prev=$loss
  if [ "1" == "$(awk "BEGIN{print($loss_new<$loss);}")" ]; then
    loss=$loss_new
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
    mv $mlp_next $mlp_best
    echo "nnet accepted ($(basename $mlp_best))"
    echo $mlp_best > $dir/.mlp_best 
  else
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
    mv $mlp_next $mlp_reject
    echo "nnet rejected ($(basename $mlp_reject))"
  fi
 
  # create .done file as a mark that iteration is over
  touch $dir/.done_iter$iter
 
  # stopping criterion
  if [[ "1" == "$halving" && "1" == "$(awk "BEGIN{print(($loss_prev-$loss)/$loss_prev < $end_halving_impr)}")" ]]; then
    if [[ "$min_iters" != "" ]]; then
      if [ $min_iters -gt $iter ]; then
        echo we were supposed to finish, but we continue, min_iters : $min_iters
        continue
      fi
    fi
    echo $iter+1ho finished, too small rel. improvement $(awk "BEGIN{print(($loss_prev-$loss)/$loss_prev)}")
    break
  fi
 
  # start annealing when improvement is low
  if [ "1" == "$(awk "BEGIN{print(($loss_prev-$loss)/$loss_prev < $start_halving_impr)}")" ]; then
    halving=1
    echo $halving >$dir/.halving
  fi
  
  # do annealing
  if [ "1" == "$halving" ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    echo $learn_rate >$dir/.learn_rate
  fi

  ## when use bn, we should decrease the learn rate 
  #decrease_factor=0.4
  #learn_rate=$(awk "BEGIN{print($learn_rate*$decrease_factor)}")
  #echo $learn_rate >$dir/.learn_rate

done
 
 ## select the best network
 if [ $mlp_best != $mlp_init ]; then 
   mlp_final=${mlp_best}_final_
   ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
   ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nostatistic.nnet; )
   echo "Succeeded training the Neural Network : $dir/final.nostatistic.nnet"
 else
   "Error training neural network..."
   exit 1
 fi
}

## at the end of train, we should get the statistic information of training data for batch normalization
statistic_tool="/styx/home/hzwuw2014/Angel-0.2/bin/nnet-train-statistic-bn"
$statistic_tool --cross-validate=true \
 --learn-rate=$learn_rate --momentum=$momentum \
 --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size \
 --randomize=false --verbose=$verbose \
 --binary=false --num-sentence=$num_sentence_tr \
 ${feature_transform:+ --feature-transform=$feature_transform} \
 "$feats_tr" "$labels_tr" "$dir/final.nostatistic.nnet" "$dir/final.nnet" \
 &> $dir/log/stastistic.tr.log || exit 1

# then, we'll evaluate the val dataset
$train_tool --cross-validate=true \
 --learn-rate=$learn_rate --momentum=$momentum \
 --minibatch-size=$(awk "BEGIN{print($minibatch_size*8)}") --randomizer-size=$(awk "BEGIN{print($randomizer_size*2)}") \
 --randomize=false --verbose=$verbose \
 --binary=false --use-trick-4test=false \
 --num-sentence=$num_sentence_cv \
 --iter=$iter+1 \
 ${feature_transform:+ --feature-transform=$feature_transform} \
 "$feats_cv" "$labels_cv" "$dir/final.nnet" \
 &> $dir/log/final.val.log 

