#include "Instance.h"
#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include <omp.h>

using namespace std;

template<class T>
void shuffle( vector<T>& vect ){
	
	int r;
	for(int i=0;i<vect.size()-1;i++){
		r =  (rand() % (vect.size() - i-1))+1+i ;
		swap(vect[i],vect[r]);
	}
}

/*double dot( vector<double>& v, SparseVec& x ){
	
	double sum=0.0;
	for(int i=0;i<x.size();i++){
	
		int index = x[i].first;
		double value = x[i].second;

		sum += v[index]*value;
	}
	
	return sum;
}*/

void trainSVM(vector<SparseVec>& data, vector<int>& labels, int N, int D, double C, double pos_weight,  vector<double>& w){
	
	//initialization
	vector<double> alpha;
	alpha.resize(N);
	
	for(int i=0;i<D;i++)
		w[i] = 0;
	for(int i=0;i<N;i++){
		alpha[i] = 0;
	}
	
	//Compute diagonal of Q matrix
	vector<double> Qii;
	Qii.resize(N);
	
	for(int i=0;i<N;i++){
		
		SparseVec& xi = data[i];
		double yi = (double) labels[i];
		
		Qii[i] = 0;
		for(int j=0;j<xi.size();j++){
			double value = xi[j].second;
			Qii[i] += value*value;
		}
	}
	
	//Main Loop
	vector<int> index;
	for(int i=0;i<N;i++)
		index.push_back(i);
	shuffle(index);

	int max_iter = 1000;
	int iter=0;
	while(iter < max_iter){
		
		for(int r=0;r<N;r++){
			
			int i = index[r];
			double yi = (double) labels[i];
			SparseVec& xi = data[i];
			double Ci = (yi>0.0)? C*pos_weight : C;
			
			//1. compute gradient of i 
			double gi = yi*dot(w,xi) - 1.0;
			//2. compute alpha_new
			double new_alpha = min( max( alpha[i] - gi/Qii[i] , 0.0 ) , Ci);
			//3. maintain v (=w)
			double alpha_diff = new_alpha-alpha[i];
			if(  fabs(alpha_diff) > 1e-12 ){
				
				int ind;
				double val;
				for(int k=0;k<xi.size();k++){
					
					ind = xi[k].first;
					val = xi[k].second;
					
					w[ind] += alpha_diff * (yi*val);
				}
				
				alpha[i] = new_alpha;
			}
		}

		//if(iter%10==0)
		//	cerr << ".";
		
		shuffle(index);
		iter++;
	}
	//cerr << endl;
}
