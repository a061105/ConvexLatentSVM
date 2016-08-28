#!/bin/bash

train_data=../SimulatedData/docs
test_data=../SimulatedData/docs


./convexTrain $train_data 10 0.001 0
../LatentSVMtrain/latentTrain $train_data 10.0 1 0 1.0 beta_assign
../LatentSVMtrain/predict $test_data model
