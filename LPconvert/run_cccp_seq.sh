#!/bin/bash

#train_data=../RealData/Seq/yst09-vs-mus/data.train
#test_data=../RealData/Seq/yst09-vs-mus/data.test
train_data=../RealData/Seq/sim_data.train
test_data=../RealData/Seq/sim_data.test

rm -f tmp2;
for i in $(seq 1 100)
do
	../LatentSVMtrain/latentTrain $train_data 10 40 1 1.0 > tmp 2> tmp
	../LatentSVMtrain/predict $train_data model 2>> tmp2 >> tmp2
	../LatentSVMtrain/predict $test_data model 2>> tmp2 >> tmp2
	python takeMax.py tmp2 'N=200, acc='
done
