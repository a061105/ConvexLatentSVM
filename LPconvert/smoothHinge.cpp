#include "../util.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>

using namespace std;

const double eta = 0.1;

double smoothHingeLoss( double z, int y ){

	double tmp = y*z;
	if( tmp > 1.0 )
		return 0.0;
	else if( tmp <= 1.0-eta )
		return 1.0-tmp-eta/2.0;
	else
		return 0.5/eta*(1.0-tmp)*(1.0-tmp);
}

double hingeLoss( double z, int y ){
	
	double tmp = y*z;
	if( tmp > 1.0 )
		return 0.0;
	else
		return 1.0 - tmp;
}

void readSol( char* sol_file, map< pair<int,int>, SparseVec >& omega, int T ){
	
	ifstream fin( sol_file );
	string line;
	while( 1 ){
		
		getline( fin, line );
		if( fin.eof() )
			break;
		
		if( line.substr(0,5)=="omega" ){
			vector<string> tokens = split(line,"\t");
			double val = atof(tokens[1].c_str());
			if( val == 0.0 )
				continue;
			
			vector<string> indexes = split( tokens[0], "-" );
			int i = atoi(indexes[1].c_str());
			int j = atoi(indexes[2].c_str());
			int h = atoi(indexes[3].c_str());
			int h2 = atoi(indexes[4].c_str());
			
			omega[ make_pair(i,j) ].push_back( make_pair(h*T+h2, val) );
		}
	}
	fin.close();
}

int main(int argc, char** argv){
	
	if( argc < 1+3 ){
		cerr << "./smoothHinge [data] [sol] [lambda]" << endl;
		exit(0);
	}

	char* data_file = argv[1];
	char* sol_file = argv[2];
	double lambda = atof(argv[3]);
	
	vector<int> labels;
	vector<Document> docs;
	readData(data_file, docs, labels);
	int m = docs.size();
	int T = docs[0].size();
	
	map< pair<int,int> , SparseVec > omega;
	readSol( sol_file, omega, T );
	
	//compute loss
	double hinge_sum = 0.0;
	double smooth_hinge_sum = 0.0;
	for(int i=0;i<m;i++){
		if( labels[i] == -1 )
			continue;
		int yi = labels[i];

		double pred_i = 0.0;
		for(int j=0;j<m;j++){
			
			int yj = labels[j];
			SparseVec& omega_ij = omega[ make_pair(i,j) ];
			for(SparseVec::iterator it=omega_ij.begin(); it!=omega_ij.end(); it++){
				
				int h = it->first/T;
				int h2 = it->first%T;
				double val = it->second;
				if( val == 0.0 )
					continue;

				double kv = kernel(docs[i][h], docs[j][h2]);
				double coeff = yj*kv/(m*lambda);
				
				pred_i += coeff*val;
			}
		}
		
		smooth_hinge_sum += smoothHingeLoss(pred_i, yi);
		hinge_sum += hingeLoss( pred_i, yi );
		
		cerr << "i=" << i << ", pred=" << pred_i<< endl;
		cerr << "i=" << i << ", loss=" << hingeLoss(pred_i, yi)<< endl;
	}

	cerr << "smooth hinge loss=" << smooth_hinge_sum << endl;
	cerr << "hinge loss=" << hinge_sum << endl;
}
