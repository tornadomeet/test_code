
#gmmDir=exp/tri2b

gmmDir=gmm_train/exp/tri2b
aliDir=exp/tri2b_ali

inputDir=gmm_train/data/train
langDir=gmm_train/lang/train

dnnFeatType=fbank

if [ ! "$gmmDir" ];then
	echo "begin train gmm baseline"
	Align_beam=10
	Align_retryBeam=25
	monoGauss=2000
	numSenone=7000
	numGauss=120000
	myscript/train_gmm.sh $monoGauss $numSenone $numGauss $Align_beam $Align_retryBeam $langDir || exit 1;
	gmmDir=exp/tri2b
fi


 #if had get fbank feature, we'll commit these
 #if [ $dnnFeatType = "fbank" ];then
	#echo "fbank"
	#myscript/compute_feat.sh $dnnFeatType train $inputDir
 #fi

local/train_dnn.sh $dnnFeatType $langDir $gmmDir $aliDir


