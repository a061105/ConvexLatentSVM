#!/bin/bash

given_h=meme.given_h

train_data=../SimulatedData/Seq/yst09r_hm08r_train.doc
test_data=../SimulatedData/Seq/yst09r_hm08r_test.doc
#train_data=../SimulatedData/Seq/RandomSeq/hm08r_train.doc
#test_data=../SimulatedData/Seq/RandomSeq/hm08r_test.doc
#given_h=../SimulatedData/Seq/RandomSeq/train_seq20nn.motif_pos
#train_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.train
#test_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.test

../LatentSVMtrain/latentTrain $train_data 100 1 1 $given_h
../LatentSVMtrain/predict $test_data model
