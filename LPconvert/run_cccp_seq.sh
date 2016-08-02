#!/bin/bash

train_data=../SimulatedData/Seq/yst08r_hm08r_train.doc
test_data=../SimulatedData/Seq/yst08r_hm08r_test.doc
#train_data=../SimulatedData/Seq/RandomSeq/hm08r_train.doc
#test_data=../SimulatedData/Seq/RandomSeq/hm08r_test.doc
#given_h=../SimulatedData/Seq/RandomSeq/train_seq20nn.motif_pos
#train_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.train
#test_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.test

rm -f tmp;
for i in $(seq 1 100)
do
	../LatentSVMtrain/latentTrain $train_data 10 40 1 #$given_h
	../LatentSVMtrain/predict $train_data model 2>> tmp >> tmp
	../LatentSVMtrain/predict $test_data model 2>> tmp >> tmp
	tail -2 tmp
done
