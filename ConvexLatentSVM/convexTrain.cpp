#include <iostream>
#include <cmath>
#include <map>
#include "../util.h"
using namespace std;
typedef multimap<double,int, greater<double> > SortKerMap;

const double TOL = 1e-4;
const double eta = 0.2;
const double tau = 5.0; //softmax parameter

const double S = 1.0;

void simplex_proj( vector<double> v, vector<double>& v_proj, int d ){
	
	vector<int> index;
	index.resize(d);
	for(int i=0;i<d;i++)
		index[i] = i;
	sort(index.begin(), index.end(), ScoreComp(&v)); //descending order
	
	int j=0;
	double part_sum = 0.0;
	for(;j<d;j++){
		int k = index[j];
		part_sum += v[k];
		if( v[k] - (part_sum-S)/(j+1) <= 0.0 ){
			part_sum -= v[k];
			j -= 1;
			break;
		}
	}
	if( j == d ) j--;
	
	double theta = (part_sum-S)/(j+1);
	
	for(int k=0;k<d;k++)
		v_proj[k] = max( v[k]-theta, 0.0);
}


void simplex_ineq_proj( vector<double> v, vector<double>& v_proj, int d ){
	
	double sum = 0.0;
	for(int i=0;i<d;i++){
		v[i] = max(v[i],0.0);
		sum += v[i];
	}
	if( sum <= S ){
		v_proj = v;
		return ;
	}
	
	vector<int> index;
	index.resize(d);
	for(int i=0;i<d;i++)
		index[i] = i;
	sort(index.begin(), index.end(), ScoreComp(&v)); //descending order
	
	int j=0;
	double part_sum = 0.0;
	for(;j<d;j++){
		int k = index[j];
		part_sum += v[k];
		if( v[k] - (part_sum-S)/(j+1) <= 0.0 ){
			part_sum -= v[k];
			j -= 1;
			break;
		}
	}
	if( j == d ) j--;
	
	double theta = (part_sum-S)/(j+1);
	
	for(int k=0;k<d;k++)
		v_proj[k] = max( v[k]-theta, 0.0);
}

double loss( double z, int y ){

	double tmp = y*z;
	if( tmp > 1.0 )
		return 0.0;
	else if( tmp <= 1.0-eta )
		return 1.0-tmp-eta/2.0;
	else
		return 0.5/eta*(1.0-tmp)*(1.0-tmp);
}

double loss_deriv( double z, int y ){

	double tmp = y*z;
	if( tmp > 1.0 )
		return 0.0;
	else if( tmp <= 1.0-eta )
		return -y;
	else
		return (1.0-tmp)*(-y)/eta;
}

typedef map< pair<int,int>, SparseVec > OmegaActMap;

class GDMMsolve{
	
	public:
	int m;
	int pos_size;
	int neg_size;
	int num_h_total;
	int num_h_neg;

	int voc_size;
	int dim;

	vector<Document> documents;
	double lambda;
	double rho;
	vector<int> labels;
	//primal variables (all 0)
	OmegaActMap omega_act;
	map< pair<int,int>, vector<double> > omega;
	map< int, SparseVec > alpha_act;
	map< int, vector<double> > alpha;
	map< int, SparseVec > beta_act;
	map< int, vector<double> > beta;
	
	GDMMsolve(char* input_file, double _lambda, double _rho){
		
		readData(input_file,    documents, labels );
		m = documents.size();
		voc_size = wordIndMap.size();
		
		lambda = _lambda*m;
		rho = _rho/m;
		
		if( kernel == BOW_kernel ){
			dim = voc_size;
		}else if( kernel == PSWM_kernel ){
			dim = voc_size*documents[0][0].size();
		}
		
		//collect positive indexes
		for(int i=0;i<m;i++)
			if( labels[i] == +1 )
				pos_index.push_back(i);
			else
				neg_index.push_back(i);
		for(int i=0;i<m;i++)
			index.push_back(i);
		
		pos_size = pos_index.size();
		neg_size = neg_index.size();
		cerr << "m=" << m << endl;
		cerr << "|pos|=" << pos_size << endl;
		cerr << "|neg|=" << neg_size << endl;
		cerr << "voc_size=" << voc_size << endl;
		cerr << "dim=" << dim << endl;
		
		//compute number of hidden slots in total
		num_h_total=0;
		num_h_neg=0;
		for(int i=0;i<m;i++){
			int Ti = documents[i].size();
			num_h_total += Ti;
			if( labels[i]==-1 )
				num_h_neg += Ti;
		}
		
		//precompute feature phi(i,j,h,h')
		Phi.resize(m);
		for(int i=0;i<m;i++){
			int Ti = documents[i].size();
			for(int h=0;h<Ti;h++)
				Phi[i].push_back(feaVect(documents[i][h]));
		}

		//precompute kernel(i,j,h,h') and sort its value for each i,j
		R_sq = 0.0;
		for(int i=0;i<m;i++){
			int Ti = documents[i].size();
			for(int j=0;j<m;j++){
				int Tj = documents[j].size();
				SortKerMap ksortMap;
				vector<double> kmat;
				for(int h=0;h<Ti;h++){
					for(int h2=0;h2<Tj;h2++){
						double kv = kernel(documents[i][h], documents[j][h2] );
						
						ksortMap.insert(make_pair(kv, h*Tj+h2));
						kmat.push_back(kv);
						//cerr << kv << " ";
						double kv_sq=kv*kv;
						if( kv_sq > R_sq )
							R_sq = kv_sq;
					}
				}

				kernel_sorted_map[ make_pair(i,j) ] = ksortMap;
				kernel_map[ make_pair(i,j) ] = kmat;
			}
		}
	}
	
	double softPredict( vector<double>& w, vector<SparseVec>& Phi_i, vector<double>* grad=NULL ){
		
		//compute inner prod for all h
		int H = Phi_i.size();
		vector<double> prod;
		prod.resize(H);
		for(int h=0;h<H;h++)
			prod[h] = tau*dot(w,Phi_i[h]);
		
		//find max
		double max_val = -1e300;
		for(int h=0;h<H;h++)
			if( prod[h] > max_val )
				max_val = prod[h];
		
		//find softmax
		double smax = 0.0;
		for(int h=0;h<H;h++){
			double tmp= exp(prod[h]-max_val);
			if( grad != NULL )
				(*grad)[h] = tmp;
			smax += tmp;
		}
		if( grad != NULL ){
			for(int h=0;h<H;h++)
				(*grad)[h] /= smax;
		}
		
		smax = (log(smax) + max_val);
		smax /= tau;
		
		return smax;
	}
	
	/*double softPredict( map<int,SparseVec>& alpha_act, vector<SparseVec>& Phi_i, int i ){
		
		int H = Phi_i.size();
		double smax = 0.0;
		for(int h=0;h<H;h++){
			
			double ker_dot = 0.0;
			for(int j=0;j<m;j++){
				vector<double>& kmat = kernel_map[ make_pair(i,j) ];
				int Tj = documents[j].size();
				SparseVec& alpha_act_j = alpha_act[j];
				for(SparseVec::iterator it=alpha_act_j.begin(); it!=alpha_act_j.end(); it++)
					ker_dot += kmat[ h*Tj + it->first ]*it->second*labels[j];
			}
			ker_dot /= lambda;
			
			smax += exp( ker_dot );
		}
		smax = log(smax);

		return smax;
	}*/
	
	void solve(){
		
		//initialize primal, dual variables
		initialize();
		cerr << "init AL_obj=" << AL_obj() << endl;
		
		//Augmented Lagrangian Loop
		int max_iter = 1000;
		int iter = 0;
		vector<double> omega_new, alpha_new, beta_new;
		while( iter < max_iter ){

			//minimmize w.r.t. omega (using FC-FW)
			random_shuffle( pos_index.begin(), pos_index.end() );
			random_shuffle( index.begin(), index.end() );
			for(int r=0;r<pos_index.size();r++){
				int i = pos_index[r];
				vector<double>& beta_i = beta[i];
				
				for(int r2=0;r2<m;r2++){
					int j = index[r2];
					int Tj = documents[j].size();
					vector<double>& alpha_j = alpha[j];
					SparseVec& omega_ij_act = omega_act[ make_pair(i,j) ];
					vector<double>& omega_ij = omega[ make_pair(i,j) ];
					vector<double>& mu_ij = mu[ make_pair(i,j) ];
					vector<double>& nu_ij = nu[ make_pair(i,j) ];
					
					//cerr << "z[" << i << "]=" << z[i] << endl;
					double loss_deriv_i = loss_deriv( z[i], labels[i] );//negative
					double yj = (double) labels[j];
					SortKerMap& ksortMap = kernel_sorted_map[ make_pair(i,j) ];
					vector<double>& kmat = kernel_map[ make_pair(i,j) ];
					//find FW direction
					double kv;
					int new_hh2 = -1;
					if( yj==1.0 ){
					   SortKerMap::iterator it;
					   for(it=ksortMap.begin(); it!=ksortMap.end(); it++)
					      if( omega_ij[it->second] == 0.0 && mu_ij[it->second]==0.0 && nu_ij[it->second]==0.0){
					         kv = it->first;
						 new_hh2 = it->second;
						 break;
					      }
					}else{ //yj==-1
					   SortKerMap::reverse_iterator it;
					   for(it=ksortMap.rbegin(); it!=ksortMap.rend(); it++)
					      if( omega_ij[it->second] == 0.0 && mu_ij[it->second]==0.0 && nu_ij[it->second]==0.0){
					         kv = it->first;
						 new_hh2 = it->second;
						 break;
					      }
					}
					if( new_hh2!=-1 )
						omega_ij_act.push_back(make_pair(new_hh2,0.0));
					
					//subsolve w.r.t. active set
					int act_size = omega_ij_act.size();
					double Qij = (1.0/eta)*act_size*R_sq/lambda/lambda + 2.0*rho;
					////compute value before proj
					omega_new.resize( act_size );
					double min_grad = 1e300;
					int argmin;
					for(int r=0;r<act_size;r++){
						int o_ind = omega_ij_act[r].first;
						double o_val = omega_ij_act[r].second;
						int h = o_ind / Tj;
						int h2 = o_ind % Tj;

						double grad = loss_deriv_i*kmat[ o_ind ]*yj/lambda 
							+ rho*max( o_val-alpha_j[h2]+ mu_ij[o_ind], 0.0 ) 
							+ rho*max( o_val-beta_i[h] + nu_ij[o_ind], 0.0 );

						omega_new[r] = o_val - grad / Qij;
						if( grad < min_grad ){
							argmin = r;
							min_grad = grad;
						}
					}
					
					if( new_hh2!=-1 && argmin != act_size-1 ){
						omega_ij_act.pop_back();
						omega_new.pop_back();
						act_size--;
					}
					
					//simplex_ineq_proj( omega_new, omega_new, act_size );
					simplex_proj( omega_new, omega_new, act_size );
					
					//maintain response z_i
					for(int r=0;r<act_size;r++){
						int o_ind = omega_ij_act[r].first;
						double o_val = omega_ij_act[r].second;
						double o_val_new = omega_new[r];

						z[i] += (o_val_new-o_val)*kmat[o_ind]*yj/lambda;
					}
					//update omega and shrink active set (remove those omega=0 and mu=0)
					SparseVec omega_ij_act_new;
					for(int r=0;r<act_size;r++){
						int o_ind = omega_ij_act[r].first;
						double o_val_new = omega_new[r];
						omega_ij[o_ind] = o_val_new;
						if( fabs(o_val_new)>TOL || fabs(mu_ij[o_ind])> TOL || fabs(nu_ij[o_ind])>TOL ){
							omega_ij_act_new.push_back(make_pair(o_ind,o_val_new));
						}else{
							omega_ij[o_ind] = 0.0;
						}
					}

					omega_ij_act = omega_ij_act_new;
				}
			}
			
			//minimize w.r.t. beta (FC-FW)
			random_shuffle(pos_index.begin(), pos_index.end());
			for(int r=0;r<pos_index.size();r++){
				int i = pos_index[r];
				SparseVec& beta_i_act = beta_act[i];
				vector<double>& beta_i = beta[i];
				int Ti = documents[i].size();
				//Find FW direction (by looking for omega!=0 or nu!=0)
				////compute grad for all h
				vector<double> grad;
				vector<int> nnz;
				grad.resize(Ti,0.0);
				for(int j=0;j<m;j++){
					SparseVec& omega_ij_act = omega_act[make_pair(i,j)];
					int Tj = documents[j].size();
					vector<double>& nu_ij = nu[make_pair(i,j)];

					for(SparseVec::iterator it=omega_ij_act.begin();
							it!=omega_ij_act.end(); it++){
						int o_ind = it->first;
						double o_val = it->second;
						int h = o_ind / Tj;
						grad[h] += -max( o_val-beta_i[h]+nu_ij[o_ind] , 0.0 );
					}
				}
				for(int h=0;h<Ti;h++)
					grad[h] *= rho;
				///find h w/ minimum grad
				double min_grad=1e300;
				int min_h;
				
				for(int h=0;h<Ti;h++)
					if( grad[h] < min_grad ){
						min_grad = grad[h];
						min_h = h;
					}
				if( beta_i[min_h] == 0.0  )
					beta_i_act.push_back( make_pair(min_h,0.0) );

				//subsolve w.r.t. active set
				////compute value before proj
				int act_size = beta_i_act.size();
				beta_new.resize(act_size);
				double Qih = rho*num_h_total;
				for(int r=0;r<act_size;r++){
					int h = beta_i_act[r].first;
					double b_val = beta_i_act[r].second;
					beta_new[r] = b_val - grad[h]/Qih;
				}
				////projection
				simplex_ineq_proj( beta_new, beta_new, act_size );
				//simplex_proj( beta_new, beta_new, act_size );
				
				//update beta and shrink active set (remove those beta=0)
				SparseVec beta_i_act_new;
				for(int r=0;r<act_size;r++){
					int b_ind = beta_i_act[r].first;
					double b_val_new = beta_new[r];
					beta_i[b_ind] = b_val_new;
					if( fabs(b_val_new)>TOL ){
						beta_i_act_new.push_back(make_pair(b_ind,b_val_new));
					}else{
						beta_i[b_ind] = 0.0;
					}
				}
				beta_i_act = beta_i_act_new;
			}
			
			//minimize w.r.t. alpha (FC-FW, needs considering negative examples)
			random_shuffle( index.begin(), index.end() );
			for(int r2=0;r2<m;r2++){
				int j = index[r2];
				SparseVec& alpha_j_act = alpha_act[j];
				vector<double>& alpha_j = alpha[j];
				int Tj = documents[j].size();
				double yj = (double) labels[j];
				//Find FW direction (by looking for omega!=0 or nu!=0)
				////compute grad: augmented terms
				vector<double> grad;
				grad.resize(Tj,0.0);
				for(int r=0;r<m;r++){
					int i = pos_index[r];
					SparseVec& omega_ij_act = omega_act[make_pair(i,j)];
					int Ti = documents[i].size();
					vector<double>& mu_ij = mu[make_pair(i,j)];

					for(SparseVec::iterator it=omega_ij_act.begin();
							it!=omega_ij_act.end(); it++){
						int o_ind = it->first;
						double o_val = it->second;
						int h2 = o_ind % Tj;
						grad[h2] += -max( o_val-alpha_j[h2]+mu_ij[o_ind] , 0.0 );
					}
				}
				for(int h2=0;h2<Tj;h2++)
					grad[h2] *= rho;
				
				////compute grad: loss of negative examples
				vector<double> feaSum, prob_i;
				feaSum.resize(dim, 0.0);
				for(int r=0;r<neg_index.size();r++){
					int i = neg_index[r];
					int Ti = documents[i].size();
					
					//compute pred_ih = w*phi(i,h) for all h
					prob_i.resize(Ti);
					z[i] = softPredict( w, Phi[i], &prob_i );
					
					//evalute loss_deriv_i
					double loss_deriv_i = loss_deriv( z[i], labels[i] );

					//collect feaVect sum
					for(int h=0;h<Ti;h++){
						double tmp = loss_deriv_i * prob_i[h];
						for(SparseVec::iterator it=Phi[i][h].begin(); it!=Phi[i][h].end(); it++)
							feaSum[it->first] += tmp * it->second;
					}
				}
				for(int h2=0;h2<Tj;h2++)
					grad[h2] += yj*dot(feaSum, Phi[j][h2])/lambda;
				
				
				///find h w/ minimum grad
				double min_grad=1e300;
				int min_h2;
				for(int h2=0;h2<Tj;h2++)
					if( grad[h2] < min_grad ){
						min_grad = grad[h2];
						min_h2 = h2;
					}
				if( alpha_j[min_h2]==0.0 )
					alpha_j_act.push_back( make_pair(min_h2,0.0) );
				
				//subsolve w.r.t. active set
				////compute value before proj
				int act_size = alpha_j_act.size();
				double Qih2 = rho*num_h_neg + eta*tau*neg_size*R_sq/lambda/lambda*act_size;
				alpha_new.resize(act_size);
				for(int r=0;r<act_size;r++){
					int h2 = alpha_j_act[r].first;
					double a_val = alpha_j_act[r].second;
					alpha_new[r] = a_val - grad[h2]/Qih2;
				}
				////projection
				//simplex_ineq_proj( alpha_new, alpha_new, act_size );
				simplex_proj( alpha_new, alpha_new, act_size );
				
				
				//update alpha and shrink active set (remove those alpha=0)
				SparseVec alpha_j_act_new;
				for(int r=0;r<act_size;r++){
					int h2 = alpha_j_act[r].first;
					
					double new_val = alpha_new[r];
					double old_val = alpha_j_act[r].second;
					
					//update alpha
					if( fabs(new_val)>TOL ){
						alpha_j_act_new.push_back(make_pair(h2,new_val));
					}else{
						new_val = 0.0;
					}
					alpha_j[h2] = new_val;
					
					//update w
					double tmp = yj*(new_val-old_val)/lambda;
					for(SparseVec::iterator it=Phi[j][h2].begin(); it!=Phi[j][h2].end(); it++)
						w[ it->first ] += tmp*it->second;
				}
				alpha_act[j] = alpha_j_act_new;
			}
			
			
			//update dual variables
			for(OmegaActMap::iterator it=omega_act.begin(); it!=omega_act.end(); it++){
				int i = it->first.first;
				int j = it->first.second;
				int Tj = documents[j].size();
				vector<double>& mu_ij = mu[ make_pair(i,j) ];
				vector<double>& nu_ij = nu[ make_pair(i,j) ];
				vector<double>& alpha_j = alpha[j];
				vector<double>& beta_i = beta[i];
				
				SparseVec& omega_act_ij = it->second;
				for( SparseVec::iterator it2=omega_act_ij.begin(); it2!=omega_act_ij.end(); it2++){
					int h_h2 = it2->first;
					int h = h_h2/Tj;
					int h2 = h_h2 % Tj;
					mu_ij[h_h2] += max( it2->second - alpha_j[h2] , 0.0);
					nu_ij[h_h2] += max( it2->second - beta_i[h], 0.0 );
				}
			}
			
			if(iter%10==0)
				cerr << "iter=" << iter << ", |A_omega|=" << average_act_size(omega_act) << ", |A_alpha|=" << average_act_size(alpha_act) << ", |A_beta|=" << average_act_size(beta_act) << ", obj=" << objective() << ", p_inf=" << primal_infeas() << endl;
				//cerr << "iter=" << iter << ", AL=" << AL_obj() << endl;
			iter++;
		}
	}
	
	private:
	vector<int> index;
	vector<int> pos_index;
	vector<int> neg_index;
	vector<vector<SparseVec> > Phi;
	map<pair<int,int>, SortKerMap > kernel_sorted_map;
	map<pair<int,int>, vector<double> > kernel_map;
	double R_sq; // an upper bound on the kernel value

	//dual variables
	map< pair<int,int>, vector<double> > mu; //the active sets of mu, nu are merged into omega_act
	map< pair<int,int>, vector<double> > nu;
	//maintain response z[i]= \frac{1}{lambda}\sum_h\sum_j\sum_h' omega(i,j,h,h') y_j K(...)
	vector<double> z; 
	//maintain w = \frac{1}{lambdaa}\sum_{j,h2} alpha(j,h2) yj phi(xj,h2)
	vector<double> w;

	void initialize(){
		for(int i=0;i<m;i++){
			alpha_act[i] = SparseVec();
			alpha[i] = vector<double>();
			int Ti = documents[i].size();
			alpha[i].resize(Ti,0.0);
			//alpha[i][0] = 1.0;
			//alpha_act[i].push_back(make_pair(0,1.0));
		}
		for(int r=0;r<pos_index.size();r++){
			int i=pos_index[r];

			beta_act[i] = SparseVec();
			beta[i] = vector<double>();
			int Ti = documents[i].size();
			beta[i].resize(Ti,0.0);
			//beta[i][0] = 1.0;
			//beta_act[i].push_back(make_pair(0,1.0));
		}
		for(int r=0;r<pos_index.size();r++){
			int i = pos_index[r];
			int Ti = documents[i].size();
			for(int j=0;j<m;j++){
				int Tj = documents[j].size();
				omega_act[ make_pair(i,j) ] = SparseVec();
				omega[ make_pair(i,j) ] = vector<double>();
				omega[ make_pair(i,j) ].resize( Ti*Tj, 0.0 );
				//omega[ make_pair(i,j) ][0] = 1.0;
				//omega_act[ make_pair(i,j) ].push_back(make_pair(0,1.0));
			}
		}

		for(int r=0;r<pos_index.size();r++){
			int i = pos_index[r];
			int Ti = documents[i].size();
			for(int j=0;j<m;j++){
				int Tj = documents[j].size();
				mu[ make_pair(i,j) ] = vector<double>();
				mu[ make_pair(i,j) ].resize( Ti*Tj, 0.0 );
				nu[ make_pair(i,j) ] = vector<double>();
				nu[ make_pair(i,j) ].resize( Ti*Tj, 0.0 );
			}
		}
		
		//maintain w = \frac{1}{lambdaa}\sum_{j,h2} alpha(j,h2) yj phi(xj,h2)
		w.resize(dim,0.0);
		
		/*for(int j=0;j<m;j++){
			double yj = labels[j];
			for(SparseVec::iterator it=Phi[j][0].begin(); it!=Phi[j][0].end(); it++)
				w[ it->first ] += it->second/lambda*yj;
		}*/
		
		
		//maintain response z[i]= \frac{1}{lambda}\sum_h\sum_j\sum_h' omega(i,j,h,h') y_j K(...)
		z.resize(m,0.0);
		for(int i=0;i<m;i++){
			if( labels[i]==1 ){
				/*for(int j=0;j<m;j++){
					double yj = (double)labels[j];
					z[i] += 1.0*yj*kernel_map[make_pair(i,j)][0]/lambda;
					//cerr << "z[" << i << "]=" << z[i] << endl;
				}*/
			}else{
				z[i] = softPredict(w, Phi[i]); //not 0
			}
		}
	}

	double average_act_size( OmegaActMap& omega_act ){

		OmegaActMap::iterator it;
		double avg=0.0;
		for(it=omega_act.begin(); it!=omega_act.end(); it++){
			avg += it->second.size();
		}
		avg /= omega_act.size();
		return avg;
	}

	double average_act_size( map<int, SparseVec>& block_act ){

		map< int, SparseVec>::iterator it;
		double avg=0.0;
		for(it=block_act.begin(); it!=block_act.end(); it++){
			avg += it->second.size();
		}
		avg /= block_act.size();
		return avg;
	}

	double objective(){
		
		vector<double> pred;
		pred.resize( m, 0.0 );
		//pred on positive examples
		OmegaActMap::iterator it;
		for(it=omega_act.begin(); it!=omega_act.end(); it++){
			pair<int,int> p = it->first;
			int i = p.first;
			int j = p.second;
			double yj = (double) labels[j];
			SparseVec& omega_ij_act = it->second;
			vector<double>& kmat = kernel_map[make_pair(i,j)];
			double tmp = yj/lambda;
			for(SparseVec::iterator it2=omega_ij_act.begin(); it2!=omega_ij_act.end(); it2++){
				pred[i] += tmp*kmat[ it2->first ]*it2->second;
			}
		}

		//pred on negative examples
		for(int r=0;r<neg_index.size();r++){
			
			int i = neg_index[r];
			int Ti = documents[i].size();
			pred[i] = softPredict( w, Phi[i] );
			//pred[i] = softPredict( alpha_act, Phi[i], i );
		}
		
		//sum up
		double sum = 0.0;
		for(int i=0;i<m;i++){
		//for(int i=0;i<pos_size;i++){
			//cerr << "pred[" << i << "]=" << pred[i] << endl;
			double tmp = loss( pred[i], labels[i] );
			sum += tmp;
		}

		return sum;
	}


	double AL_obj(){
		
		double obj = objective();

		double AL_alpha_term = 0.0;
		double AL_beta_term = 0.0;
		OmegaActMap::iterator it;
		for(it=omega_act.begin(); it!=omega_act.end(); it++){

			pair<int,int> p = it->first;
			int i = p.first;
			int j = p.second;
			SparseVec& omega_ij_act = it->second;

			int Tj = documents[j].size();
			vector<double>& alpha_j = alpha[j];
			vector<double>& beta_i = beta[i];
			vector<double>& mu_ij = mu[make_pair(i,j)];
			vector<double>& nu_ij = nu[make_pair(i,j)];
			
			for(SparseVec::iterator it2=omega_ij_act.begin(); it2!=omega_ij_act.end(); it2++){
				int h = it2->first % Tj;
				int h2 = it2->first % Tj;
				double tmp = max( it2->second - alpha_j[h2] + mu_ij[it2->first], 0.0 );
				AL_alpha_term += tmp*tmp;
				double tmp2 = max( it2->second - alpha_j[h2] + mu_ij[it2->first], 0.0 );
				AL_beta_term += tmp2*tmp2;
			}
		}
		AL_alpha_term *= rho/2.0;
		AL_beta_term *= rho/2.0;
		
		//cerr <<"obj=" << obj << ", AL_a=" << AL_alpha_term << ", AL_b=" << AL_beta_term << endl;
		return obj + AL_alpha_term + AL_beta_term;
		//return obj + AL_alpha_term;
	}

	double primal_infeas(){
		
		double max_infeas = -1e300;
		for(OmegaActMap::iterator it=omega_act.begin(); it!=omega_act.end(); it++){
			int i = it->first.first;
			int j = it->first.second;
			int Tj = documents[j].size();
			vector<double>& mu_ij = mu[ make_pair(i,j) ];
			vector<double>& nu_ij = nu[ make_pair(i,j) ];
			vector<double>& alpha_j = alpha[j];
			vector<double>& beta_i = beta[i];

			SparseVec& omega_act_ij = it->second;
			for( SparseVec::iterator it2=omega_act_ij.begin(); it2!=omega_act_ij.end(); it2++){
				int h_h2 = it2->first;
				int h = h_h2/Tj;
				int h2 = h_h2 % Tj;
				double infeas_a = max( it2->second - alpha_j[h2] , 0.0);
				double infeas_b = max( it2->second - beta_i[h], 0.0 );
				if( infeas_a > max_infeas )
					max_infeas = infeas_a;
				if( infeas_b > max_infeas )
					max_infeas = infeas_b;
			}
		}
		return max_infeas;
	}
};
	
int main(int argc, char** argv){

		if( argc < 1+3 ){
			cerr << "./ConvexTrain [data] [lambda] [rho]"  << endl;
			exit(0);
		}

		char* input_file = argv[1];
		double lambda = atof(argv[2]);
		double rho = atof(argv[3]);
		
		GDMMsolve solver(input_file, lambda, rho);
		
		solver.solve();
		
		map<int, SparseVec>& beta_act = solver.beta_act;
		for(int i=0;i<solver.labels.size();i++){
			if( solver.labels[i]==1 ){
				cout << i << " ";
				SparseVec& beta_act_i = beta_act[i];
				sort(beta_act_i.begin(), beta_act_i.end(), PairValueComp());
				for( SparseVec::iterator it=beta_act_i.begin(); it!=beta_act_i.end(); it++)
					cout << it->first << ":" << it->second << " ";
				
				cout << endl;
			}
		}
		
		/*map<pair<int,int>, SparseVec>& omega_act = solver.omega_act;
		int num_pos = solver.pos_size;
		int num = solver.m;
		int T = solver.documents[0].size();
		for(int i=0;i<num_pos;i++){
			for(int j=0;j<num;j++){
				SparseVec& omega_act_ij = omega_act[make_pair(i,j)];
				cout << "omega-" << i << "-" << j << "= " ;
				for( SparseVec::iterator it=omega_act_ij.begin(); it!=omega_act_ij.end(); it++)
					cout << "(" << (it->first/T) << "," << (it->first%T) << "):" << it->second << " ";
				cout << endl;
			}
		}*/
		
		return 0;
}
