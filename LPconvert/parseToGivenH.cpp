#include "../util.h"
#include <iostream>
using namespace std;

const int MAX_NUM_DOC = 10000;

void readBeta( char* fname, vector<vector<double> >& beta){
	
	beta.resize(MAX_NUM_DOC);
	
	ifstream fin(fname);
	string var_name;
	double score;
	int last_doc_id = -1;
	while( !fin.eof() ){

		fin >> var_name >> score;
		if( var_name.substr(0,4) == "beta" ){
			vector<string> triple;
			triple = split( var_name, "-" );
			int doc_id = atoi(triple[1].c_str());
			int sen_id = atoi(triple[2].c_str());
			
			beta[doc_id].push_back( score );
		}
	}
}

int argmax( vector<double>& v, int& max_ind ){
	
	double max_val = -1e300;
	for(int i=0;i<v.size();i++)
		if( v[i] > max_val ){
			max_val = v[i];
			max_ind = i;
		}

	return max_ind;
}

int main(int argc, char** argv){
	
	if( argc < 1+1 ){
		cerr << "./parseToGivenH [var_sol]" << endl;
		cerr << "Output: [var_sol].given_h" << endl;
		exit(0);
	}
	
	char* var_sol_fname = argv[1];
	vector<vector<double> > beta;
	
	readBeta(var_sol_fname, beta);
	
	//for each doc, find sen of largest beta
	vector<int> h_list;
	for(int i=0;i<beta.size();i++){
		if( beta[i].size() == 0 )
			break;
		int h;
		argmax( beta[i], h );
		h_list.push_back(h);
	}
	
	//write h-list to file
	string suffix = ".given_h";
	string output_fname = string(var_sol_fname) + suffix;
	ofstream fout( output_fname.c_str() );
	for(int i=0;i<h_list.size();i++)
		fout << h_list[i] << endl;
	fout.close();
	
	return 0;
}
