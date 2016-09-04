#!/bin/bash

train_data=../SimulatedData/docs.train
test_data=../SimulatedData/docs.test
#train_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.train
#test_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.test

given_h=../SimulatedData/keypos.train

rm -f tmp2;
for i in $(seq 1 1000)
do
	../LatentSVMtrain/latentTrain $train_data 10.0 40 0 > tmp 2> tmp #$given_h 
	../LatentSVMtrain/predict $train_data model 2>> tmp2 >> tmp2
	../LatentSVMtrain/predict $test_data model 2>> tmp2 >> tmp2
	python takeMax.py tmp2 'N=200, acc='
done

#../LatentSVMtrain/latentTrain $train_data 10.0 40 0
#../LatentSVMtrain/predict $test_data model
