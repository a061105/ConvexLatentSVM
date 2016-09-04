#!/bin/bash

dir=../RealData/Seq/
train_data=$dir/sim_data.train
test_data=$dir/sim_data.test
#train_data=../RealData/Seq/yst09-vs-mus/data.train
#test_data=../RealData/Seq/yst09-vs-mus/data.test

num_tighten_iter=1
feature_option=1
rho=0.01
lambda=10

C=10.0
cccp_iter=40

#./convexTrain $train_data  $lambda  $rho  $feature_option  $num_tighten_iter
for t in $(seq 0 ${num_tighten_iter}) 
do
	../LatentSVMtrain/latentTrain -h beta_assign.t${t} $train_data  $C  1  $feature_option
	../LatentSVMtrain/predict $test_data model
	../LatentSVMtrain/latentTrain -h beta_assign.t${t} $train_data  $C  $cccp_iter  $feature_option
	../LatentSVMtrain/predict $test_data model
done

#../LatentSVMtrain/latentTrain -h $dir/motif_pos.train $train_data $C 1 $feature_option
