#!/bin/bash

dir=../RealData/Seq/
train_data=$dir/sim_data.train
test_data=$dir/sim_data.test
#train_data=../RealData/Seq/yst09-vs-mus/data.train
#test_data=../RealData/Seq/yst09-vs-mus/data.test

./convexTrain $train_data 10 0.01 1
../LatentSVMtrain/latentTrain -h beta_assign $train_data 10.0 40 1
#../LatentSVMtrain/latentTrain -h $dir/motif_pos.train $train_data 10.0 1 1
#../LatentSVMtrain/latentTrain -w model_init $train_data 10.0 2 1
../LatentSVMtrain/predict $test_data model
