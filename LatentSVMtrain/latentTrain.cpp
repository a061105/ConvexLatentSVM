#include "latentSVM.h"

void readGivenH(char* fname, vector<int>& h){
	
	ifstream fin(fname);
	if( fin.fail() ){
		cerr << "file " << fname << " not found." << endl;
		exit(0);
	}
	int doc_ind, h_val;
	
	while( !fin.eof() ){
		fin >> doc_ind  >> h_val;
		if( !fin.eof() ){
			h[ doc_ind ] = h_val;
		}
	}
}

int main(int argc, char** argv){
	
	if( argc < 1+5 ){
		cerr << "./latentSVM [train_doc] [C] [#iter] [fea_option] [pos_weight] (given_h)" << endl;
		cerr << "output: model" << endl;
		cerr << "feature options:" << endl;
		cerr << "	0: bag-of-word" << endl;
		cerr << "	1: position-specific weight matrix" << endl;
		exit(0);
	}
	
	char* train_doc_fname = argv[1];
	double C = atof( argv[2] );
	int nIter = atoi( argv[3] );
	int fea_option = atoi( argv[4] );
	double pos_weight = atof( argv[5] );
	char* given_h_fname = NULL;
	if( argc > 1+5 )
		given_h_fname = argv[6];
	
	srand(time(NULL));
	vector<Document> docs;
	vector<int> labels;
	readData( train_doc_fname, docs, labels );
	
	/*vector<string> wordMap;
	wordMap.resize( wordIndMap.size() );
	for(map<string,int>::iterator it=wordIndMap.begin(); it!=wordIndMap.end(); it++)
		wordMap[it->second] = it->first;
	for(int i=0;i<docs[0].size();i++){
		for(int j=0;j<docs[0][i].size();j++) 
			cerr << wordMap[docs[0][i][j]] << " ";
		cerr << endl;
	}
	cerr << endl;*/

	int N = docs.size();
	cerr << "num of docs=" << N << endl;
	int voc_size = wordIndMap.size();
	cerr << "|voc|=" << voc_size << endl;
	
	int dim;
	if( fea_option == 0 ){
		dim = voc_size;
		feaVect = BOWfeaVect;
	}else if( fea_option == 1 ){
		feaVect = PSWMfeaVect;
		dim = voc_size*docs[0][0].size();
	}else{
		cerr << "[error]: No such feature option: " << fea_option << endl;
		exit(0);
	}
	cerr << "dim=" << dim << endl;

	vector<double> w;
	w.resize(dim, 0.0);
	
	vector<int> h;
	h.resize( N );
	for(int i=0;i<N;i++){
		if( labels[i] == 1 )
			h[i] = rand()%( docs[i].size() );
	}
	if( given_h_fname != NULL ){
		readGivenH( given_h_fname , h );
		/*for(int i=0;i<5;i++){
			int s = h[i];
			for(int j=0;j<docs[i][s].size();j++){
				cerr << wordMap[ docs[i][s][j] ];
			}
			cerr << endl;
		}
		exit(0);*/
	}
	
	vector<SparseVec> data_svm;
	vector<int> labels_svm;
	for(int iter=0; iter<nIter; iter++){
		cerr << "iter=" << iter << endl;
		// Given h, solve w
		//// Generate xi for i being positive
		data_svm.clear();
		labels_svm.clear();
		int N_pos = 0;
		for(int i=0;i<N;i++){
			if( labels[i]==-1 )
				continue;
			N_pos++;
			SparseVec xi = feaVect( docs[i][ h[i] ] );
			data_svm.push_back(xi);
			labels_svm.push_back(+1);
		}
		//// Generate xi for i being negative
		int N_neg=0;
		for(int i=0;i<N;i++){
			if( labels[i]==1 )
				continue;
			int Ti = docs[i].size();
			for(int j=0;j<Ti;j++){
				N_neg++;
				SparseVec xi = feaVect( docs[i][j] );
				data_svm.push_back(xi);
				labels_svm.push_back(-1);
			}
		}
		
		//// Use xi, yi to train w
		trainSVM(data_svm, labels_svm, N_pos+N_neg, dim, C, pos_weight,  w);
		//for(int i=0;i<voc_size;i++)
		//	cout << w[i] << endl;
		
		// Given w, solve h for positive examples
		for(int i=0;i<N;i++){
			if( labels[i] == -1 )
				continue;
			
			int Ti = docs[i].size();
			double max_dot = -1e300;
			int argmax;
			for(int j=0;j<Ti;j++){
				SparseVec xij = feaVect(docs[i][j]);
				double dot_ij = dot( w, xij );
				if( dot_ij > max_dot ){
					max_dot = dot_ij;
					argmax = j;
				}
			}
			h[i] = argmax;
		}
	}
	
	writeModel("model", w, fea_option);
	cout << "train acc=" << accuracy( docs, labels, w ) << endl;
}
