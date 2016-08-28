#!/bin/bash

train_data=../SimulatedData/docs.train
test_data=../SimulatedData/docs.test
#train_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.train
#test_data=../RealData/ChineseOpinion/parsed_docs/state-3.op-as-pos.test


./LPconvert $train_data 10 0
../LPsparse/LPsparse -e 0.01 -t 5e-2 .
./paste.sh
./parseToGivenH var_sol

../LatentSVMtrain/latentTrain $train_data 1.0 1 0  1.0 var_sol.given_h
../LatentSVMtrain/predict $test_data model
