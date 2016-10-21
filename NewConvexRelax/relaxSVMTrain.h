#include "../util.h"

typedef vector<int> Atom;
typedef vector<pair<double, Atom> > AtomCombin;

/**
 * Interface Relax SVM is responsibple for the training of SVM 
 * when "a linear combination of latent-variables is given".
 */ 
class RelaxSVM{
	
	public:
	/**Main function
	 * INPUT: a linear combination of latent indexes, 
	 * 		each index vector is of length N.
	 * OUTPUT: optimal dual variables given the latent.
	 */
	virtual void relaxTrain(AtomCombin& M, vector<double>& alpha)=0;
};

/* Positive loss: alpha_N and single
 * Negative loss: beta_NH per atom
 */
class SmoothHingeSVM:public RelaxSVM{
	
	public:
	SmoothHingeSVM(vector<Document>& data, vector<int>& labels, int D, double C, double rho){
		
		N = labels.size();
		for(int i=0;i<N;i++){
			if( labels[i] > 0 ){
				data_pos[i] = data[i];
			}else{
				data_neg[i] = data[i];
			}
		}
		Np = data_pos.size();
		Nn = data_neg.size();
		Hp.resize(Np);
		for(int i=0;i<Np;i++)
			Hp[i] = data_pos[i].size();
		Hn.resize(Nn);
		for(int i=0;i<Nn;i++)
			Hn[i] = data_neg[i].size();
		
		this->D = D;
		this->C = C;
		this->rho = rho;
	}
	
	void relaxTrain( AtomCombin& M, vector<double>& alpha){
		
		//initialize dual/primal variables
		int K = M.size(); //atomic support size
		
		alpha.resize(Np, 0.0);
		beta.resize(K);
		for(int i=0;i<K;i++){
			beta[i].resize(Nn);
			for(int j=0;j<Nn;j++)
				beta[i][j].resize(Hn[j], 0.0);
		}
		w.resize(K);
		for(int i=0;i<K;i++)
			w[i].resize( D, 0.0 );
		
		//Compute diagonal of Q matrix
		vector<double> Qii_pos;
		Qii_pos.resize(Np, 0.0);
		for(int k=0;k<K;k++){
			double mu_k = M[k].first;
			Atom& h_k = M[k].second;
			for(int i=0;i<Np;i++){
				SparseVec& xi = data_pos[i][h_k[i]];
				Qii_pos[i] += mu_k*norm_sq(xi);
			}
		}
		for(int i=0;i<Np;i++)
			Qii_pos[i] += rho;

		vector<vector<double> > Qii_neg;//without timing mu_k
		Qii_neg.resize(Nn);
		for(int i=0;i<Nn;i++)
			Qii_neg.resize(Hn[i]);
		for(int i=0;i<Nn;i++){
			for(int h=0; h<Hn[i]; h++){
				SparseVec& x_ih = data_neg[i][h];
				Qii_neg[i][h] = norm_sq(x_ih);
			}
		}
		
		//Main Loop
		int max_iter = 1000;
		int iter=0;
		while(iter < max_iter){
			
			vector<int> index = random_perm( Np+Nn*K );
			for(int r=0; r<Np+Nn*K; r++){
				
				if( index[r] < Np ){ //Positive Examples

					int i = index[r];
					double alpha_i = alpha[i];
					//1. compute gradient of i 
					double gi = -1.0;
					for(int k=0;k<K;k++){
						double mu_k = M[k].first;
						int h_ki = M[k].second[i];
						SparseVec& x_ih = data_pos[i][h_ki];
						
						gi += mu_k*dot(w[k], x_ih);
					}
					gi += rho*alpha_i;
					
					//2. compute alpha_new
					double new_alpha_i = min( max( alpha_i - gi/Qii_pos[i] , 0.0 ) , C);
					//3. update w and alpha
					double alpha_diff_i = new_alpha_i - alpha_i;
					for(int k=0;k<K;k++){
						int h_ki = M[k].second[i];
						vadd( w[k], alpha_diff_i, data_pos[i][h_ki] );
					}
					
					alpha[i] = new_alpha_i;

				}else{//Negative Examples

					int k = (index[r] - Np)/Nn; //remove offset
					int i = (index[r] - Np)%Nn;
					double yi = -1.0;
					double mu_k = M[k].first;
					
					vector<int> subIndex = random_perm( Hn[i] );
					for(vector<int>::iterator it = subIndex.begin();
							it != subIndex.end(); it++){
						
						int h = *it;
						double beta_kih = beta[k][i][h];
						//1. compute gradient of alpha_ih for all h
						SparseVec& x_ih = data_neg[i][h];
						double g = mu_k*yi*dot(w[k],x_ih) - 1.0 + rho*beta_kih;
						
						//2. compute update
						double Q_kih = Qii_neg[i][h]*mu_k + rho;
						double new_beta = min( max( beta_kih - g/Q_kih , 0.0 ) , C);
						
						//3. update w and beta
						double beta_diff = new_beta - beta_kih;
						vadd( w[k], yi*beta_diff, data_neg[i][h] );
						
						beta[k][i][h] = new_beta;
					}
				}
			}
			
			iter++;
		}
	}

	private:
	
	/* Optimization variables
	 */
	vector<double> alpha;
	vector<vector<vector<double> > > beta;
	vector<vector<double> > w;
	
	/* Inputs
	 */
	int N;
	int Np;
	int Nn;
	vector<Document> data_pos;
	vector<int> Hp;
	vector<Document> data_neg;
	vector<int> Hn;
	int D;
	double C;
	double rho; //smoothing parameter for hinge loss
};
