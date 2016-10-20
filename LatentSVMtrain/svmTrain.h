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

void trainSVM(vector<SparseVec>& data_pos, vector<vector<SparseVec> >& data_neg, vector<int>& labels_org, int D, double C, double pos_weight,  vector<double>& w){
	
	int N=0, Np=data_pos.size();
	vector<SparseVec> data;
	vector<int> labels;
	for(int i=0;i<data_pos.size();i++){
		data.push_back( data_pos[i] );
		labels.push_back( labels_org[i] );
		N++;
	}
	for(int i=0;i<data_neg.size();i++){
		int yi = labels_org[Np+i];
		for(int h=0;h<data_neg[i].size();h++){
			data.push_back( data_neg[i][h] );
			labels.push_back( yi );
			N++;
		}
	}
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

	int max_iter = 100;
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

/** Train a hidden-variable SVM with hidden variables given on positive samples (a convex problem).
 *  min_{w} \sum_{i\in pos} L( w^Tphi(x_i,h_i) ) + \sum_{i\in neg} L( \max_{h} w^Tphi(x_i,h) )
 */
void trainHiddenSVM(vector<SparseVec>& data_pos, vector<vector<SparseVec> >& data_neg, vector<int>& labels, int D, double C,  vector<double>& w){
	
	//initialization
	int Np = data_pos.size();
	int Nn = data_neg.size();
	vector<double> alpha_pos;
	alpha_pos.resize(Np);
	vector<vector<double> > alpha_neg;
	alpha_neg.resize(Nn);
	for(int i=0;i<Nn;i++)
		alpha_neg[i].resize(data_neg[i].size());
	
	for(int i=0;i<D;i++)
		w[i] = 0;
	for(int i=0;i<Np;i++)
		alpha_pos[i] = 0;
	for(int i=0;i<Nn;i++){
		for(int h=0;h<alpha_neg[i].size();h++)
			alpha_neg[i][h] = 0.0;
	}
	
	//Compute diagonal of Q matrix
	vector<double> Qii_pos;
	Qii_pos.resize(Np);
	for(int i=0;i<Np;i++){
		SparseVec& xi = data_pos[i];
		Qii_pos[i] = norm_sq(xi);
	}
	
	vector<double> Qii_neg;
	Qii_neg.resize(Nn);
	for(int i=0;i<Nn;i++){
		double max_Q = -1e300;
		for(int h=0; h<data_neg[i].size(); h++){
			SparseVec& x_ih = data_neg[i][h];
			double Qihih = norm_sq(x_ih);
			if( Qihih > max_Q )
				max_Q = Qihih;
		}
		Qii_neg[i] = max_Q;
	}
	
	//Main Loop
	vector<int> index;
	for(int i=0;i<Np+Nn;i++)
		index.push_back(i);
	shuffle(index);
	
	int max_iter = 1000;
	int iter=0;
	while(iter < max_iter){
		
		for(int r=0;r<Np+Nn;r++){
			
			int i = index[r];
			double yi = (double) labels[i];
			
			if( yi > 0.0 ){
				SparseVec& xi = data_pos[i];
				//1. compute gradient of i 
				double gi = yi*dot(w,xi) - 1.0;
				//2. compute alpha_new
				double new_alpha = min( max( alpha_pos[i] - gi/Qii_pos[i] , 0.0 ) , C);
				//3. update w and alpha
				double alpha_diff = new_alpha-alpha_pos[i];
				vadd( w, yi*alpha_diff, xi );
				alpha_pos[i] = new_alpha;
			
			}else{
				i = i - Np; //remove offset
				int Hi = data_neg[i].size();
				double Qii = Qii_neg[i];
				
				vector<double> alpha_i_new;
				alpha_i_new.resize(Hi);
				//1. compute gradient of alpha_ih for all h
				for(int h=0;h<Hi;h++){
					SparseVec& xih = data_neg[i][h];
					double grad_ih = yi*dot(w,xih) - 1.0;
					alpha_i_new[h] = alpha_neg[i][h]-grad_ih/Qii;
				}
				//2. compute simplex (inequality) projection
				simplex_ineq_proj( alpha_i_new, alpha_i_new, Hi, C);
				
				//3. update w and alpha
				for(int h=0;h<Hi;h++){
					double alpha_ih_diff = alpha_i_new[h] - alpha_neg[i][h];
					vadd( w, yi*alpha_ih_diff, data_neg[i][h] );
					alpha_neg[i][h] = alpha_i_new[h];
				}
			}
		}

		shuffle(index);
		iter++;
	}
	//cerr << endl;
}
