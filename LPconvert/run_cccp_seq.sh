#!/bin/bash

#train_data=../RealData/Seq/yst09-vs-mus/data.train
#test_data=../RealData/Seq/yst09-vs-mus/data.test
train_data=../RealData/Seq/yst09-vs-mus/data.train
test_data=../RealData/Seq/yst09-vs-mus/data.test

rm -f tmp;
for i in $(seq 1 100)
do
	../LatentSVMtrain/latentTrain $train_data 10 40 1 1.0 #$given_h
	../LatentSVMtrain/predict $train_data model 2>> tmp >> tmp
	../LatentSVMtrain/predict $test_data model 2>> tmp >> tmp
	tail -1 tmp
done
