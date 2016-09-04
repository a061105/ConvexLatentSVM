#include "latentSVM.h"

int main(int argc, char** argv){
	
	if( argc < 1+2 ){
		cerr << "./predict [data] [model]" << endl;
		exit(0);
	}
	
	char*  datafname = argv[1];
	char*  modelfname = argv[2];
	
	vector<double> w;
	//model must be read before data to ensure wordIndMap is constructed by model
	readModel( modelfname, w );
	vector<Document> docs;
	vector<int> labels;
	readData( datafname,  docs,  labels );
	int N = docs.size();
	//cerr << "num of docs=" << N << endl;
	
	cerr << "N=" << N << ", acc=" << accuracy( docs, labels, w ) << endl;
	
	ofstream fout("output");
	for(int i=0;i<N;i++){
		fout << predict( w, docs[i] ) << endl;
	}
	fout.close();
}
