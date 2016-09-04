#!/bin/bash


#train_data=../RealData/Seq/yst09-vs-mus/data.train
#test_data=../RealData/Seq/yst09-vs-mus/data.test

train_data=../RealData/Seq/sim_data.train
test_data=../RealData/Seq/sim_data.test

./LPconvert $train_data 10 1
../LPsparse/LPsparse -e 0.01 -t 0.05 .
./paste.sh
./parseToGivenH var_sol

../LatentSVMtrain/latentTrain -h var_sol.given_h $train_data 10.0 1 1
../LatentSVMtrain/predict $test_data model
