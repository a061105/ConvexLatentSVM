#include <iostream>
#include "../util.h"
#include "Instance.h"
#include "svmTrain.h"

using namespace std;

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

void writeModel(char* fname, vector<double>& w, int fea_option){

	ofstream fout(fname);
	fout << "feature_type: " <<  fea_option << endl;
	fout << w.size() << endl;
	for(int i=0;i<w.size();i++)
		fout << w[i] << " ";
	fout << endl;
	
	map<string,int>::iterator it;
	for(it=wordIndMap.begin(); it!=wordIndMap.end(); it++)
		fout << it->first << " " << it->second << endl;

	fout.close();
}

void readModel(char* fname, vector<double>& w){
	
	ifstream fin(fname);
	if( fin.fail() ){
		cerr << "fail to read " << fname << endl;
		fin.close();
	}
	
	string tmp;
	int fea_type;
	fin >> tmp >> fea_type;
	if( fea_type == 0 ){
		feaVect = BOWfeaVect;
	}else if( fea_type == 1 ){
		feaVect = PSWMfeaVect;
	}else{
		cerr << "[error]: No such feature option: " << fea_type << endl;
		exit(0);
	}

	int D;
	fin >> D;
	w.resize(D);
	double wi;
	for(int i=0;i<D;i++){
		fin >> wi;
		w[i] = wi;
	}
	
	string word;
	int ind;
	for(int i=0;i<D;i++){
		fin >> word >> ind;
		wordIndMap[word] = ind;
	}
	fin.close();
}
