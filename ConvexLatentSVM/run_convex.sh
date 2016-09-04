#!/bin/bash

dir=../SimulatedData/
train_data=$dir/docs.train
test_data=$dir/docs.test


#./convexTrain $train_data 10 0.01 0
#../LatentSVMtrain/latentTrain -h beta_assign $train_data 10.0 40 0 
../LatentSVMtrain/latentTrain -h $dir/keypos.train $train_data 10.0 1 0 
#../LatentSVMtrain/latentTrain -w model_init $train_data 10.0 2 0 
../LatentSVMtrain/predict $test_data model
