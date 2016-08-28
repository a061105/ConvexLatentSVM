#!/bin/bash


#train_data=../RealData/Seq/yst09-vs-mus/data.train
#test_data=../RealData/Seq/yst09-vs-mus/data.test

train_data=../RealData/Seq/sim_data.train
test_data=../RealData/Seq/sim_data.test

./LPconvert $train_data 100 1
../LPsparse/LPsparse -e 0.01 -t 0.01 .
./paste.sh
./parseToGivenH var_sol

../LatentSVMtrain/latentTrain $train_data 10 1 1 10.0 var_sol.given_h
../LatentSVMtrain/predict $test_data model
