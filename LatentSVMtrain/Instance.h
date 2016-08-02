#ifndef INSTANCE
#define INSTANCE

#include<vector>
#include<iostream>

using namespace std;

typedef vector<pair<int,double> > SparseVec;
	
void printSparseVec( ostream& out, SparseVec& sv ){
	
	for(SparseVec::iterator it=sv.begin(); it!=sv.end(); it++){
		out << it->first << ":" << it->second << endl;
	}
}
#endif
