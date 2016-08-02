#!/bin/bash

train_data=../SimulatedData/train_doc.40.noise
test_data=../SimulatedData/test_doc.40.noise
#train_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.train
#test_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.test


rm -f tmp2;
for i in $(seq 1 100)
do
	../LatentSVMtrain/latentTrain $train_data 10 40 0 #$given_h
	../LatentSVMtrain/predict $train_data model 2>> tmp2 >> tmp2
	../LatentSVMtrain/predict $test_data model 2>> tmp2 >> tmp2
	tail -2 tmp2
done

#../LatentSVMtrain/latentTrain $train_data 10.0 40 0
#../LatentSVMtrain/predict $test_data model
