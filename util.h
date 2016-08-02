#include <fstream>
#include <map>
#include <limits.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cassert>
using namespace std;

typedef vector<int> Sentence;
typedef vector<Sentence> Document;
typedef vector<pair<int,double> > SparseVec;

class ScoreComp{
	public:
		vector<double>* score;
		ScoreComp(vector<double>* _score){
			score = _score;
		}
		bool operator()(const int& ind1, const int& ind2){
			return (*score)[ind1] > (*score)[ind2];
		}
};

class PairValueComp{
	public:
		bool operator()(const pair<int,double>& it1, const pair<int,double>& it2){
			return it1.second > it2.second;
		}
};

void print( ostream& out, SparseVec& v ){
	
	for(SparseVec::iterator it=v.begin(); it!=v.end() ;it++)
		out << it->first << ":" << it->second << " " ;
	out << endl;
}

void print( ostream& out, vector<double>& v ){
	
	for(vector<double>::iterator it=v.begin(); it!=v.end() ;it++)
		out << *it << " " ;
	out << endl;
}

SparseVec::iterator argmax( SparseVec& v ){
	
	double max_val = -1e300;
	SparseVec::iterator ret;
	for(SparseVec::iterator it=v.begin(); it!=v.end(); it++){
		if( it->second > max_val ){
			ret = it;
			max_val = it->second;
		}
	}
	return ret;
}

double dot(vector<double>& v, SparseVec& v2){
	
	double sum = 0.0;
	for(SparseVec::iterator it=v2.begin(); it!=v2.end(); it++){
		sum += v[it->first]*it->second;
	}
	return sum;
}

int max_of( vector<int>& v ){
	
	int max_val = -INT_MIN;
	for(vector<int>::iterator it=v.begin(); it!=v.end(); it++)
		if( *it > max_val )
			max_val = *it;

	return max_val;
}

SparseVec vadd( SparseVec& v1, SparseVec& v2 ){

	SparseVec v3;
	SparseVec::iterator it1 = v1.begin(); 
	SparseVec::iterator it2 = v2.begin();
	while( it1!=v1.end() && it2!=v2.end() ){
		
		if( it1->first < it2->first ){
			v3.push_back(make_pair(it1->first, it1->second));
			it1++;
		}else if( it2->first < it1->first ){
			v3.push_back(make_pair(it2->first, it2->second));
			it2++;
		}else{
			int ind = it1->first;
			double val = it1->second + it2->second;
			v3.push_back(make_pair(ind, val));
			it1++;
			it2++;
		}
	}
	
	if( it1==v1.end() ){
		for(; it2!=v2.end(); it2++)
			v3.push_back(make_pair(it2->first, it2->second));
	}else{
		for(; it1!=v1.end(); it1++)
			v3.push_back(make_pair(it1->first, it1->second));
	}
	
	return v3;
}

vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		string sub = str.substr(i,index-i);
		if(sub.length()>0 && sub!=" ")
			str_split.push_back(sub);
		
		i = index+1;
	}
	
	//if( str_split.back()=="" )
	//	str_split.pop_back();

	return str_split;
}

map<string,int> wordIndMap;
vector<string> wordMap;
int getIndex( string& token ){
	
	if( wordIndMap.find(token) == wordIndMap.end() ){
		wordIndMap.insert( make_pair(token, wordIndMap.size()) );
	}
	
	return wordIndMap[token];
}

Sentence parse_sentence( string token ){
	
	Sentence sen;

	vector<string> tokens = split( token, " " );
	for(int i=0;i<tokens.size();i++){
		
		if( tokens[i] == "" )
			continue;
		
		int word = getIndex( tokens[i] );
		sen.push_back(word);
	}

	return sen;
}

void readData(char* input_fname,  vector<Document>& documents, vector<int>& labels){

	ifstream fin(input_fname);
	string line;
	
	//allocate number of documents
	getline( fin, line );
	int nDoc = atoi(line.c_str());
	for(int i=0;i<nDoc;i++)
		documents.push_back(Document());
	
	for(int i=0;i<nDoc;i++){
		
		getline( fin,  line );
		//get label
		vector<string> tokens = split(line, ", ");
		int label = atoi(tokens[0].c_str());
		labels.push_back(label);
		
		//split doc
		string doc = tokens[1];
		tokens = split(doc, ".");
		for(int j=0;j<tokens.size();j++)
			documents[i].push_back( parse_sentence(tokens[j]) );
	}
	
	fin.close();

	//build word map
	wordMap.resize(wordIndMap.size());
	for(map<string,int>::iterator it=wordIndMap.begin(); it!=wordIndMap.end(); it++){
		wordMap[it->second] = it->first;
	}
}


SparseVec BOWfeaVect( Sentence& sen ){
	
	sort(sen.begin(), sen.end());
	
	SparseVec phi;
	int last = -1;
	for(Sentence::iterator it=sen.begin(); it!=sen.end(); it++){
		int word = *it;
		if( word == last ){
			phi.back().second += 1.0;
		}else{
			phi.push_back( make_pair(word,1.0) );
		}
		last = word;
	}
	double sen_size_sqrt = sqrt((double)sen.size());
	for(SparseVec::iterator it=phi.begin(); it!=phi.end(); it++)
		it->second /= sen_size_sqrt;
	
	return phi;
}

SparseVec PSWMfeaVect( Sentence& sen ){
	
	SparseVec phi;
	int voc_size = wordIndMap.size();
	//double sen_len = (double) sen.size();
	//double sen_len = 1.0;
	double len = (double)sen.size();
	double sqr_len = sqrt(len);
	for(int i=0;i<sen.size();i++){
		int word = sen[i];
		phi.push_back( make_pair( i*voc_size + word, 1.0/sqr_len ) );
		//phi.push_back( make_pair( i*voc_size + word, 1.0 ) );
	}
	
	return phi;
}


double BOW_kernel(Sentence& s1, Sentence& s2){
	
	
	///SparseVec Inner Product
	sort(s1.begin(), s1.end());
	sort(s2.begin(), s2.end());
	
	int num_common=0;
	int i=0,j=0;
	while( i<s1.size() && j<s2.size() ){

		if( s1[i] == s2[j] ){
			num_common++;
			i++;
			j++;
		}else if( s1[i] < s2[j] ){
			i++;
		}else{
			j++;
		}
	}
	double s1_size = (double) s1.size();
	double s2_size = (double) s2.size();

	return  ((double)num_common/sqrt(s1_size*s2_size));
}


double PSWM_kernel(Sentence& s1, Sentence& s2){
	
	assert(s1.size()==s2.size());
	int len = s1.size();
	int count=0;
	for(int i=0;i<len;i++){
		if( s1[i]==s2[i] )
			count++;
	}
	return (double)count/len;
	//return (double)count;
}


SparseVec (*feaVect)(Sentence&) = BOWfeaVect;
//SparseVec (*feaVect)(Sentence&) = PSWMfeaVect;
double (*kernel)(Sentence&, Sentence&) = BOW_kernel;
//double (*kernel)(Sentence&, Sentence&) = PSWM_kernel;
