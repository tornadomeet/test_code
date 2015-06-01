#!/bin/bash

if [ -f path.sh ]; then . ./path.sh; fi

#compute word error rate
data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/result.txt $data/text; do
  [ ! -f $f ] && echo "computeWER.sh: no such file $f" && exit 1;
done

cat $dir/result.txt | utils/int2sym.pl -f 2- $symtab | sed 's:\<UNK\>::g' | \
compute-wer --text=true --mode=present ark,t:$data/text ark,p,t:- > $dir/wer || exit 1;

