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
	}
	
	vector<SparseVec> data_pos;
	vector<vector<SparseVec> > data_neg;
	vector<int> labels_svm;
	for(int iter=0; iter<nIter; iter++){
		cerr << "iter=" << iter << endl;
		// Given h, solve w
		//// Generate xi for i \in positive
		data_pos.clear();
		for(int i=0;i<N;i++){
			if( labels[i]==-1 )
				continue;
			SparseVec xi = feaVect( docs[i][ h[i] ] );
			data_pos.push_back(xi);
		}
		//// Generate xi for i \in negative
		for(int i=0;i<N;i++){
			if( labels[i]==1 )
				continue;
			data_neg.push_back(vector<SparseVec>());
			int Hi = docs[i].size();
			for(int j=0;j<Hi;j++){
				SparseVec xi = feaVect( docs[i][j] );
				data_neg.back().push_back(xi);
			}
		}
		
		//// Use xi, yi to train w
		trainHiddenSVM(data_pos, data_neg, labels, dim, C,  w);
		
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
