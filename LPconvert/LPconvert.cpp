#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include "../util.h"
#include <map>
#include <algorithm>
using namespace std;

struct Triple{
	int i;
	int j;
	double val;
	Triple(int _i, int _j, double _val){
		i = _i;
		j = _j;
		val = _val;
	}
};

map<string,int> id_map;

int varID( string varName ){
	
	if( id_map.find(varName) == id_map.end() ){
		int newID = id_map.size();
		id_map[ varName ] = newID;

		return newID;
	}
	else{
		return id_map[varName];
	}
}

string alpha(int i, int h){
	
	char c_str[1000];
	sprintf(c_str,"alpha-%d-%d", i, h );
	string str(c_str);
	
	return str;	
}
string beta(int i, int h){
	
	char c_str[1000];
	sprintf(c_str,"beta-%d-%d", i, h );
	string str(c_str);
	
	return str;	
}
string omega(int i, int j, int h, int h2){
	
	char c_str[1000];
	sprintf(c_str,"omega-%d-%d-%d-%d", i, j, h, h2 );
	string str(c_str);
	
	return str;	
}
string xi(int i){
	
	char c_str[1000];
	sprintf(c_str,"xi-%d", i );
	string str(c_str);
	
	return str;	
}

void writeMeta(char* fname, int m, int meq, int n){
	
	ofstream fout(fname);
	fout << "nb\t" << n << endl;
	fout << "nf\t" << 0 << endl;
	fout << "mI\t" << m << endl;
	fout << "mE\t" << meq << endl;
	fout.close();
}

void writeVect(char* outputFname, vector<double>& vec){
	
	ofstream fout( outputFname );
	for(int i=0;i<vec.size();i++){
		fout << vec[i] << endl;
	}
	fout.close();
}

void writeMat(char* outputFname, int m, int n, vector<Triple>& mat){
	
	ofstream fout( outputFname );
	fout << m << " " << n << " " << "0" << endl;
	for(int i=0;i<mat.size();i++){
		fout << mat[i].i+1 << " " << mat[i].j+1 << " " << mat[i].val << endl;
	}
	fout.close();
}

void writeVarMap( char* fname, map<string,int>& idmap ){

	vector<string> name_map;
	name_map.resize(idmap.size());
	//go through all <name, id> pair in idmap
	map<string,int>::iterator it;
	for(it=idmap.begin(); it!=idmap.end(); it++){
		string name = it->first;
		int id = it->second;
		name_map[id] = name;
	}
	
	//write to file
	ofstream fout(fname);
	for(int i=0;i<name_map.size();i++)
		fout << name_map[i] << endl;
	fout.close();
}


int main(int argc, char** argv){
	
	if( argc < 1 + 2 ){
		cerr << "./LPconvert [train_doc] [lambda] (fea_option)" << endl;
		cerr << "Output: A, b, c, Aeq, beq" << endl;
		cerr << "feature options:" << endl;
		cerr << "	0: bag-of-word" << endl;
		cerr << "	1: position-specific weight matrix" << endl;
		exit(0);
	}
	
	char* input_file = argv[1];
	double lambda = atof(argv[2]);
	int fea_option = 0;
	if( argc > 1 + 2 )
		fea_option = atoi(argv[3]);
	
	if( fea_option == 0 ){
		//do nothing; use default kernel
		kernel = BOW_kernel;
	}else if( fea_option==1 ){
		kernel = PSWM_kernel;
	}else{
		cerr << "[error:] unknown feature option " << fea_option << endl;
		exit(0);
	}
	
	vector<Document> documents;
	vector<int> labels;
	readData(input_file,    documents, labels );
	
	int m = documents.size();
	int voc_size = wordIndMap.size();
	cerr << "m=" << m << endl;
	cerr << "|voc|=" << voc_size << endl;
	
	int eqID=0, ineqID=0;
	vector<Triple> Aeq;
	vector<Triple> A;
	vector<double> beq;
	vector<double> b;
	vector<double> c;
	
	// sum_{h} alpha_{j,h} = 1, for all j
	/*for(int j=0;j<m;j++){
		//if( labels[j] == -1 )
		//	continue;

		Document doc = documents[j];
		int Tj = doc.size();
		
		for(int h=0;h<Tj;h++){
			Aeq.push_back(Triple(eqID, varID(alpha(j,h)), 1.0));
		}
		beq.push_back(1.0);
		
		eqID++;
	}*/
	
	// sum_{h2} beta_{i,h2} = 1, for i=1...m
	/*for(int i=0;i<m;i++){
		
		if( labels[i] == -1 )
			continue;

		Document doc = documents[i];
		int Ti = doc.size();
		
		for(int h2=0;h2<Ti;h2++){
			Aeq.push_back(Triple(eqID, varID(beta(i,h2)), 1.0));
		}
		beq.push_back(1.0);
		
		eqID++;
	}*/

	// sum_{h} sum_{h2} omega_{i,j,h,h2}=1; for all j and i \in \POS
	for(int i=0;i<m;i++){
		if( labels[i] == -1 )
			continue;
		
		cerr << "omega i=" << i << endl;
		Document doc1 = documents[i];
		int Ti = doc1.size();
		for(int j=0;j<m;j++){
			//if( labels[j] != 1 )
			//	continue;
			
			Document doc2 = documents[j];
			int Tj = doc2.size();
			
			// sum_{h} sum_{h2} omega_{i,j,h,h2}=1
			for(int h=0; h<Tj; h++){
				for(int h2=0;h2<Ti;h2++){
					Aeq.push_back(
						Triple(eqID, varID(omega(i,j,h,h2)), 1.0));
				}
			}
			
			beq.push_back(1.0);
			eqID++;
		}
		
		/*for(int j=0;j<m;j++){
			if( labels[j] != -1 )
				continue;
			
			Document doc2 = documents[j];
			int Tj = doc2.size();
			
			// sum_{h2} omega_{i,j,h,h2}=alpha_{j,h}, for all h
			for(int h=0; h<Tj; h++){
				for(int h2=0;h2<Ti;h2++){
					Aeq.push_back(
						Triple(eqID, varID(omega(i,j,h,h2)), 1.0));
				}
				Aeq.push_back(Triple(eqID, varID(alpha(i,h)), -1.0));
				beq.push_back(0.0);
				eqID++;
			}
		}*/
	}
	
	// omega_{i,j,h,h2} <= alpha_{j,h}
	/*for(int i=0;i<m;i++){
		if( labels[i]==-1 )
			continue;
		cerr << "omega <= alpha, i=" << i << endl;
		int Ti = documents[i].size();
		for(int j=0;j<m;j++){
			int Tj = documents[j].size();
			for(int h=0;h<Tj;h++){
				for(int h2=0;h2<Ti;h2++){
					
					A.push_back(Triple(ineqID,varID(omega(i,j,h,h2)),1.0));
					A.push_back(Triple(ineqID,varID(alpha(j,h)),-1.0));
					b.push_back(0.0);
					
					ineqID++;
				}
			}
		}
	}*/

	// omega_{i,j,h,h2} <= beta_{i,h2}
	/*for(int i=0;i<m;i++){
		if( labels[i]==-1 )
			continue;
		cerr << "omega <= beta, i=" << i << endl;
		int Ti = documents[i].size();
		for(int j=0;j<m;j++){
			int Tj = documents[j].size();
			for(int h=0;h<Tj;h++){
				for(int h2=0;h2<Ti;h2++){
					
					A.push_back(Triple(ineqID,varID(omega(i,j,h,h2)),1.0));
					A.push_back(Triple(ineqID,varID(beta(i,h2)),-1.0));
					b.push_back(0.0);
					
					ineqID++;
				}
			}
		}
	}*/
	
	//generate c
	int n = id_map.size() + m;
	c.resize(n,0.0);
	for(int i=0;i<m;i++)
		c[ varID( xi(i) ) ] = 1.0;
	
	for(int i=0;i<m;i++){
		if( labels[i] == -1 ){
			continue;
		}
		cerr << "xi+, i=" << i << endl;
		int Ti = documents[i].size();
		for(int j=0;j<m;j++){
			int Tj = documents[j].size();
			double yj = labels[j];
			for(int h=0;h<Tj;h++){
				Sentence& s1 = documents[j][h];
				for(int h2=0;h2<Ti;h2++){
					Sentence& s2 = documents[i][h2];
					double kernel_ijhh2 = yj*kernel(s1,s2)/m/lambda;
					if( kernel_ijhh2 != 0.0 )
						A.push_back(
						Triple(ineqID,varID(omega(i,j,h,h2)),-kernel_ijhh2) );
				}
			}
		}
		
		A.push_back( Triple(ineqID, varID(xi(i)), -1.0) );
		b.push_back( -1.0 );
		
		ineqID++;
	}
	
	/*for(int i=0;i<m;i++){
		if( labels[i] == 1 )
			continue;
		
		cerr << "xi-, i=" << i << endl;
		int Ti = documents[i].size();
		for(int h2=0;h2<Ti;h2++){
			Sentence& s2 = documents[i][h2];
			//generate 1 constraint
			A.push_back( Triple(ineqID, varID(xi(i)), -1.0) );
			for(int j=0;j<m;j++){
				int yj = labels[j];
				int Tj = documents[j].size();
				for(int h=0;h<Tj;h++){
					Sentence& s1 = documents[j][h];
					double kernel_ijhh2 = yj*kernel(s1,s2)/m/lambda;
					if( kernel_ijhh2 != 0.0 )
						A.push_back( Triple(ineqID,varID(alpha(j,h)),kernel_ijhh2) );
				}
			}
			b.push_back(-1.0);
			
			ineqID++;
		}
	}*/

	int m_ineq = ineqID;
	int m_eq = eqID;
	cerr << "writing..." << endl;
	
	writeMeta("meta", m_ineq, m_eq, n );
	cerr << "write A..." << endl;
	writeMat( "A", m_ineq, n, A );
	cerr << "write Aeq..." << endl;
	writeMat( "Aeq", m_eq, n, Aeq );
	writeVect( "beq", beq );
	writeVect( "b", b );
	writeVect( "c", c );
	writeVarMap( "varMap", id_map );
}
