#include "latentSVM.h"

void readGivenH(char* fname, vector<int>& h, vector<int>& pos_index ){
	
	ifstream fin(fname);
	if( fin.fail() ){
		cerr << "file " << fname << " not found." << endl;
		exit(0);
	}

	int h_val;
	int i=0;
	for(int i=0;i<pos_index.size();i++){
		fin >> h_val;
		h[pos_index[i]] = h_val;
	}
}

void exit_with_help(){

	cerr << "./latentSVM [train_doc] [C] [#iter] [fea_option]" << endl;
	cerr << "output: model" << endl;
	cerr << "optiones:" << endl;
	cerr << "	-h hidden_assign_file: initialize hidden variables w/ a given assignments." << endl;
	cerr << "	-w model_init_file: initialize model w." << endl;
	cerr << "	-p positive_weight: reweight loss of positive samples (default 1.0)." << endl;
	cerr << "feature options:" << endl;
	cerr << "	0: bag-of-word" << endl;
	cerr << "	1: position-specific weight matrix" << endl;
	cerr << "	2: linear" << endl;
	exit(0);
}

void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	for(i=1;i<argc;i++){
		if( argv[i][0] != '-' )
			break;
		if( ++i >= argc )
			exit_with_help();

		switch(argv[i-1][1]){
			
			case 'h': param->init_h_fpath = argv[i];
				  break;
			case 'w': param->init_w_fpath = argv[i];
				  break;
			case 'p': param->pos_weight = atoi(argv[i]);
				  break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
				  exit(0);
		}
	}
	
	if(i>=argc)
		exit_with_help();
	
	param->train_doc_fpath = argv[i++];
	param->C = atof( argv[i++] );
	param->nIter = atoi( argv[i++] );
	param->fea_option = atoi( argv[i++] );
}

int main(int argc, char** argv){
	
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	
	srand(time(NULL));
	
	//read model if -w is specified
	vector<double> w;
	if( param->init_w_fpath != NULL ){
		readModel( param->init_w_fpath, w );
	}
	
	//read data
	vector<Document> docs;
	vector<int> labels;
	readData( param->train_doc_fpath, docs, labels );
	vector<int> pos_index, neg_index;
	for(int i=0;i<labels.size();i++){
		if( labels[i]==1 ){
			pos_index.push_back(i);
		}else{
			neg_index.push_back(i);
		}
	}
	int N = docs.size();
	cerr << "num of docs=" << N << endl;
	cerr << "num of pos=" << pos_index.size() << endl;
	cerr << "num of neg=" << neg_index.size() << endl;
	int voc_size = wordIndMap.size();
	cerr << "|voc|=" << voc_size << endl;
	
	int dim;
	if( param->fea_option == 0 ){
		feaVect = BOWfeaVect;
		dim = voc_size;
	}else if( param->fea_option == 1 ){
		feaVect = PSWMfeaVect;
		dim = voc_size*docs[0][0].size();
	}else if( param->fea_option == 2 ){
		feaVect = linearFeaVect;
		dim = voc_size;
	}else{
		cerr << "[error]: No such feature option: " << param->fea_option << endl;
		exit(0);
	}
	cerr << "dim=" << dim << endl;

	if( param->init_w_fpath==NULL )
		w.resize(dim, 0.0);
	
	vector<int> h;
	h.resize( N );
	for(vector<int>::iterator it=pos_index.begin(); it!=pos_index.end(); it++){
		h[*it] = rand()%( docs[*it].size() );
	}
	if( param->init_h_fpath != NULL ){
		readGivenH( param->init_h_fpath , h , pos_index );
	}
	
	//// Generate xi for i \in negative
	vector<vector<SparseVec> > data_neg;
	data_neg.clear();
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

	vector<SparseVec> data_pos;
	vector<int> labels_svm;
	for(int iter=0; iter<param->nIter; iter++){
		
		//cerr << "iter=" << iter << endl;
		cerr << "#";

		// Given h, solve w
		//// Generate xi for i \in positive
		if( iter!=0 || param->init_w_fpath==NULL ){
			data_pos.clear();
			for(int i=0;i<N;i++){
				if( labels[i]==-1 )
					continue;
				SparseVec xi = feaVect( docs[i][ h[i] ] );
				data_pos.push_back(xi);
			}

			//// Use xi, yi to train w
			if( param->pos_weight < 0.0 )
				trainHiddenSVM(data_pos, data_neg, labels, dim, param->C,  w);
			else
				trainSVM(data_pos, data_neg, labels, dim, param->C, param->pos_weight, w);
		}

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
	cerr << endl;
	
	writeModel("model", w, param->fea_option);
	writeVect("h_pos", h);
	cout << "train acc=" << accuracy( docs, labels, w ) << endl;
}
