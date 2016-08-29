#!/bin/bash

train_data=../SimulatedData/docs.train
test_data=../SimulatedData/docs.test
#train_data=../../docs.train
#test_data=../../docs.test


./convexTrain $train_data 10 0.01 0
../LatentSVMtrain/latentTrain $train_data 10.0 1 0 1.0 beta_assign
../LatentSVMtrain/predict $test_data model
