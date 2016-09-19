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

typedef vector<pair<int,double> > Sentence;
typedef vector<Sentence> Document;
typedef vector<pair<int,double> > SparseVec;
const int FNAME_LEN = 1000;

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

class PairIndexComp{
	public:
		bool operator()(const pair<int,double>& it1, const pair<int,double>& it2){
			return it1.first < it2.first;
		}
};

class PairValueComp{
	public:
		bool operator()(const pair<int,double>& it1, const pair<int,double>& it2){
			return it1.second > it2.second;
		}
};

void printMap( ostream& out, map<int,int>& m ){
	
	out << "map[";
	for(map<int,int>::iterator it=m.begin(); it!=m.end(); it++){
		out << it->first << ":" << it->second << ", ";
	}
	out << "]";
}

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

void writeVect( char* fname, vector<int>& v ){
	
	ofstream fout(fname);
	for(int i=0;i<v.size();i++)
		fout << v[i] << endl;
	fout.close();
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

double norm_sq(SparseVec& v){
	
	double sum = 0.0;
	for(SparseVec::iterator it=v.begin(); it!=v.end(); it++){
		double val = it->second;
		sum += val*val;
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

void vadd(vector<double>& w, double c, SparseVec& sv){
	
	for(SparseVec::iterator it=sv.begin(); it!=sv.end(); it++)
		w[it->first] += c * it->second;
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

void simplex_proj( vector<double> v, vector<double>& v_proj, int d, double S ){
	
	vector<int> index;
	index.resize(d);
	for(int i=0;i<d;i++)
		index[i] = i;
	sort(index.begin(), index.end(), ScoreComp(&v)); //descending order
	
	int j=0;
	double part_sum = 0.0;
	for(;j<d;j++){
		int k = index[j];
		part_sum += v[k];
		if( v[k] - (part_sum-S)/(j+1) <= 0.0 ){
			part_sum -= v[k];
			j -= 1;
			break;
		}
	}
	if( j == d ) j--;
	
	double theta = (part_sum-S)/(j+1);
	
	for(int k=0;k<d;k++)
		v_proj[k] = max( v[k]-theta, 0.0);
}


void simplex_ineq_proj( vector<double> v, vector<double>& v_proj, int d, double S ){
	
	double sum = 0.0;
	for(int i=0;i<d;i++){
		v[i] = max(v[i],0.0);
		sum += v[i];
	}
	if( sum <= S ){
		v_proj = v;
		return ;
	}
	
	vector<int> index;
	index.resize(d);
	for(int i=0;i<d;i++)
		index[i] = i;
	sort(index.begin(), index.end(), ScoreComp(&v)); //descending order
	
	int j=0;
	double part_sum = 0.0;
	for(;j<d;j++){
		int k = index[j];
		part_sum += v[k];
		if( v[k] - (part_sum-S)/(j+1) <= 0.0 ){
			part_sum -= v[k];
			j -= 1;
			break;
		}
	}
	if( j == d ) j--;
	
	double theta = (part_sum-S)/(j+1);
	
	for(int k=0;k<d;k++)
		v_proj[k] = max( v[k]-theta, 0.0);
}

vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		string sub = str.substr(i,index-i);
		if(sub.length()>0 && sub!=" " && sub!="  ")
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
		
		int word;
		double val;
		if( tokens[i].find(":") == string::npos ){
			word = getIndex(tokens[i]);
			val = 1.0;
		}else{
			vector<string> ind_val = split(tokens[i],":");
			word = getIndex(ind_val[0]);
			val = atof(ind_val[1].c_str());
		}
		sen.push_back(make_pair(word, val));
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
		tokens = split(doc, ". ");
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
		int word = it->first;
		if( word == last ){
			phi.back().second += it->second;
		}else{
			phi.push_back( make_pair(word,it->second) );
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
		int word = sen[i].first;
		phi.push_back( make_pair( i*voc_size + word, sen[i].second/sqr_len ) );
		//phi.push_back( make_pair( i*voc_size + word, sen[i].second ) );
	}
	
	return phi;
}

SparseVec linearFeaVect( Sentence& sen ){
	
	return sen;
}


double BOW_kernel(Sentence& s1, Sentence& s2){//assume s1 and s2 are sorted and collapsed
	
	sort(s1.begin(), s1.end(), PairIndexComp());
	sort(s2.begin(), s2.end(), PairIndexComp());

	double prod=0.0;
	int i=0,j=0;
	while( i<s1.size() && j<s2.size() ){

		if( s1[i].first == s2[j].first ){
			prod += 1.0;
			i++;
			j++;
		}else if( s1[i].first < s2[j].first ){
			i++;
		}else{
			j++;
		}
	}
	double s1_size = (double) s1.size();
	double s2_size = (double) s2.size();
	
	return  ((double)prod/sqrt(s1_size*s2_size));
}


double PSWM_kernel(Sentence& s1, Sentence& s2){
	
	assert(s1.size()==s2.size());
	int len = s1.size();
	double count=0.0;
	for(int i=0;i<len;i++){
		if( s1[i].first==s2[i].first )
			count += 1.0;
	}
	return count/len;
	//return count;
}

double linear_kernel(Sentence& s1, Sentence& s2){//assume s1 and s2 are sorted and collapsed
	
	sort(s1.begin(), s1.end(), PairIndexComp());
	sort(s2.begin(), s2.end(), PairIndexComp());
	
	double prod=0.0;
	int i=0,j=0;
	while( i<s1.size() && j<s2.size() ){

		if( s1[i].first == s2[j].first ){
			prod += s1[i].second*s2[j].second;
			i++;
			j++;
		}else if( s1[i].first < s2[j].first ){
			i++;
		}else{
			j++;
		}
	}
	double s1_size = (double) s1.size();
	double s2_size = (double) s2.size();
	
	return  ((double)prod)/s1_size;
}


//SparseVec (*feaVect)(Sentence&) = BOWfeaVect;
SparseVec (*feaVect)(Sentence&) = PSWMfeaVect;
//double (*kernel)(Sentence&, Sentence&) = BOW_kernel;
double (*kernel)(Sentence&, Sentence&) = PSWM_kernel;

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
	}else if( fea_type == 2 ){
		feaVect = linearFeaVect;	
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
