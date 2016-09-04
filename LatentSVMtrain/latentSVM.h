#include <iostream>
#include "../util.h"
#include "Instance.h"
#include "svmTrain.h"

using namespace std;

class Param{
	public:
	char* train_doc_fpath;
	double C;
	int nIter;
	int fea_option; //for L1-norm (default 1/N)
	
	char* init_h_fpath;
	char* init_w_fpath;
	double pos_weight; //weight of loss of positive samples
	
	Param(){
		init_h_fpath = NULL;
		init_w_fpath = NULL;
		pos_weight = 1.0;
	}
};

/*SparseVec genFeature( Sentence& sen, int voc_size, int option ){
	
	vector<int> set;
	set.resize(voc_size, 0);
	
	for(int i=0;i<sen.size();i++){
		if( sen[i] < set.size() )
			set[ sen[i] ] += 1;
	}
	
	SparseVec xi;
	for(int i=0;i<set.size();i++){
		if( set[i] > 0 )
			xi.push_back(make_pair(i,set[i]));
	}
	
	return xi;
}*/


double predict(vector<double>& w, Document& doc){
	
	int voc_size = w.size();

	double max_val = -1e300;
	int argmax;
	for(int i=0;i<doc.size();i++){
		Sentence sen = doc[i];
		SparseVec xi = feaVect(sen);
		double dot_i = dot(w,xi);
		if( dot_i > max_val ){
			max_val = dot_i;
			argmax = i;
		}
	}

	return max_val;
}

double accuracy(vector<Document>& docs, vector<int>& labels, vector<double>& w){
	
	int hit=0;
	for(int i=0;i<docs.size();i++){
		
		double pred = predict(w, docs[i]);
		int yi = labels[i];
		
		if( yi*pred > 0.0 ){
			hit++;
		}
	}
	
	return ((double)hit)/docs.size();
}

