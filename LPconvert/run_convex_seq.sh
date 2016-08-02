#!/bin/bash

#train_data=../SimulatedData/Seq/train_seq.doc
#test_data=../SimulatedData/Seq/test_seq.doc

train_data=../SimulatedData/Seq/yst08r_hm080910r_train.doc
test_data=../SimulatedData/Seq/yst08r_hm080910r_test.doc
#train_data=../SimulatedData/Seq/RandomSeq/yst09r_train.doc
#test_data=../SimulatedData/Seq/RandomSeq/yst09r_test.doc
#train_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.train
#test_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.test


#./LPconvert $train_data 10 1
#../LPsparse/LPsparse -e 0.01 -t 0.01 .
#./paste.sh
#./parseToGivenH var_sol

../LatentSVMtrain/latentTrain $train_data 1 40 1 var_sol.given_h
../LatentSVMtrain/predict $test_data model
