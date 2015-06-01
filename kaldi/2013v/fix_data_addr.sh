#/bin/sh

subfeat_dir=data-fmllr-tri3b-fbank/train/_data
njob=11


for(( i=1;i<$njob;i++))
do
	echo "processing the " $i "th .scp"
	sed -i  's/mnt\/disk\/\0\/multimedia\/hzchenhaibo\/sourcecode\/kaldi-trunk\/egs\/rm\/yixin_0420_16k/home\/hzwuw2014\/kaldi_2013v\/eges\/rm\/dnn_2gpu_20150420_16k/g' $subfeat_dir"/feats_fmllr_train."$i".scp" 
done

#feat_dir=data-fmllr-tri3b-fbank/train/feats.scp
#echo "processing the feat.scp"
#sed -i  's/styx\/home\/hzchenhaibo\/sourcecode\/kaldi-trunk\/egs\/rm\/yixin_0420_16k/home\/hzwuw2014\/kaldi_2013v\/eges\/rm\/dnn_2gpu_20150420_16k/g' $feat_dir


#train_dir=exp/tri3b_pretrain-dbn_fbank/train.scp
#cv_dir=exp/tri3b_pretrain-dbn_fbank/train.scp.10k
#echo "processing the feat.scp"
#sed -i  's/styx\/home\/hzchenhaibo\/sourcecode\/kaldi-trunk\/egs\/rm\/yixin_0811/home\/hzwuw2014\/kaldi_2013v\/eges\/rm\/dnn_2gpu_20150420_16k/g'            $train_dir 
#sed -i  's/styx\/home\/hzchenhaibo\/sourcecode\/kaldi-trunk\/egs\/rm\/yixin_0811/home\/hzwuw2014\/kaldi_2013v\/eges\/rm\/dnn_2gpu_20150420_16k/g'            $cv_dir 

#train90_dir=data-fmllr-tri3b-fbank/train_tr90/feats.scp
#cv10_dir=data-fmllr-tri3b-fbank/train_cv10/feats.scp
#echo "processing the feat.scp"
#sed -i  's/mnt\/disk\/\0\/multimedia\/hzchenhaibo\/sourcecode\/kaldi-trunk\/egs\/rm\/yixin_0420_16k/home\/hzwuw2014\/kaldi_2013v\/eges\/rm\/dnn_2gpu_20150420_16k/g' $train90_dir
#sed -i  's/mnt\/disk\/\0\/multimedia\/hzchenhaibo\/sourcecode\/kaldi-trunk\/egs\/rm\/yixin_0420_16k/home\/hzwuw2014\/kaldi_2013v\/eges\/rm\/dnn_2gpu_20150420_16k/g' $cv10_dir
