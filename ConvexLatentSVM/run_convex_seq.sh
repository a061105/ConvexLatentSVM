#!/bin/bash

train_data=../RealData/Seq/sim_data.train
test_data=../RealData/Seq/sim_data.test
#train_data=../RealData/Seq/yst09-vs-mus/data.train
#test_data=../RealData/Seq/yst09-vs-mus/data.test


./convexTrain $train_data 10 0.01 1
../LatentSVMtrain/latentTrain $train_data 10.0 40 1 1.0 beta_assign
../LatentSVMtrain/predict $test_data model
