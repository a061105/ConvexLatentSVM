#!/bin/bash

dir=../SimulatedData/
train_data=$dir/docs.train
test_data=$dir/docs.test

num_tighten_iter=0
feature_option=0
rho=0.01
lambda=10

C=10.0
cccp_iter=40

./convexTrain $train_data  $lambda  $rho  $feature_option  $num_tighten_iter
for t in $(seq 0 ${num_tighten_iter}) 
do
	../LatentSVMtrain/latentTrain -h beta_assign.t${t} $train_data  $C  1  $feature_option
	../LatentSVMtrain/predict $test_data model
	../LatentSVMtrain/latentTrain -h beta_assign.t${t} $train_data  $C  $cccp_iter  $feature_option
	../LatentSVMtrain/predict $test_data model
done

#../LatentSVMtrain/latentTrain -h $dir/keypos.train $train_data $C 1 $feature_option 
