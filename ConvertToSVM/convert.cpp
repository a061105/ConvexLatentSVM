#include <stdlib.h>
#include <iostream>
#include "../util.h"

SparseVec BOWfeaVec( Document& doc ){
	
	SparseVec phi;
	for(int j=0;j<doc.size();j++){
		SparseVec phi_j = BOWfeaVect( doc[j] );
		phi = vadd( phi, phi_j );
	}
	double size = (double) doc.size();
	for(SparseVec::iterator it=phi.begin(); it!=phi.end(); it++)
		it->second /= size;
	
	return phi;
}

SparseVec avgPSWMfeaVec( Document& doc ){
	
	SparseVec phi;
	for(int j=0;j<doc.size();j++){
		SparseVec phi_j = PSWMfeaVect( doc[j] );
		phi = vadd( phi, phi_j );
	}
	double size = (double) doc.size();
	for(SparseVec::iterator it=phi.begin(); it!=phi.end(); it++)
		it->second /= size;
	
	return phi;
}

int freq_threshold = 3;
map<Sentence, int> motif_id_map;
void buildFreqMotif( vector<Document>& docs ){

	//count frequency
	map<Sentence, int> motif_freq_map;
	for(int i=0;i<docs.size();i++){
		Document& doc = docs[i];
		for(int j=0;j<doc.size();j++){

			map<Sentence,int>::iterator it;
			if(  (it=motif_freq_map.find(doc[j])) == motif_freq_map.end() ){
				motif_freq_map.insert( make_pair(doc[j],0) );
			}else{
				it->second++;
			}
		}
	}

	//if freq >= thd
	for(map<Sentence,int>::iterator it=motif_freq_map.begin(); it!=motif_freq_map.end(); it++){

		if( it->second >= freq_threshold ){

			motif_id_map.insert( make_pair(it->first, motif_id_map.size()) );
		}
	}
}
	
	
SparseVec motifCountFeaVec( Document& doc ){

	vector<int> fea_count;
	fea_count.resize( motif_id_map.size(), 0 );

	map<Sentence, int>::iterator it;
	for(int j=0;j<doc.size();j++){
		if( (it=motif_id_map.find( doc[j] )) != motif_id_map.end() ){
			fea_count[ it->second ] = 1;
		}
	}

	SparseVec phi;
	for(int i=0;i<fea_count.size();i++){
		if( fea_count[i] > 0 )
			phi.push_back(make_pair(i,fea_count[i]));
	}

	return phi;
}

int count(Document& doc, int voc_index){
	
	int _count=0;
	for(int i=0;i<doc.size();i++){
		Sentence sen = doc[i];
		for(int j=0;j<sen.size();j++){
			int word = sen[j];
			if( word == voc_index )
				_count++;
		}
	}
	
	return _count;
}


int main(int argc, char** argv){
	
	if( argc < 1+ 4){
		cerr << "Usage: ./convert [train_doc] [test_doc] [output_train_file] [output_test_file] (fea_option)" << endl;
		cerr << "fea_option:" << endl;
		cerr << "	0: average PSWM (default)" << endl;
		cerr << "       1: frequent motif count" << endl;
		cerr << "       2: bag of words" << endl;
		exit(0);
	}

	char* train_fname = argv[1];
	char* test_fname = argv[2];
	char* output_train_fname = argv[3];
	char* output_test_fname = argv[4];
	int fea_option = 0;
	if(argc > 1+4 )
		fea_option = atoi(argv[5]);
	
	SparseVec (*feaVect)(Document&);
	if( fea_option==0 ){
		feaVect = avgPSWMfeaVec;
	}else if( fea_option==1 ){
		feaVect = motifCountFeaVec;
	}else if( fea_option==2 ){
		feaVect = BOWfeaVec;
	}else{
		cerr << "no such option" << endl;
	}
	

	// read file to construct document
	vector<Document> documents;
	vector<Document> test_documents;
	vector<int> labels;
	vector<int> test_labels;
	readData( train_fname, documents, labels );
	readData( test_fname, test_documents, test_labels );
	cerr << "#train_doc=" << documents.size() << endl;
	cerr << "#test_doc=" << test_documents.size() << endl;
	if( fea_option == 1 ){
		buildFreqMotif( documents );
		ofstream fout("freq_motifs");
		for(map<Sentence,int>::iterator it=motif_id_map.begin();
				it!=motif_id_map.end(); it++){
			Sentence sen = it->first;
			int id = it->second;
			for(Sentence::iterator it2=sen.begin(); it2!=sen.end(); it2++)
				fout << wordMap[*it2] ;
			fout << "\t" << id+1 << endl;
		}
		fout.close();
	}
	//find vocSize
	int vocSize = -1;
	for(int i=0;i<documents.size();i++){
		Document doc = documents[i];
		for(int j=0;j<doc.size();j++){
			Sentence sen = doc[j];
			for(int k=0;k<sen.size();k++){
				int word = sen[k];
				if( word > vocSize )
					vocSize = word;
			}
		}
	}
	vocSize +=1;
	cerr << "vocsize=" << vocSize << endl;
	
	// convert train docs to SVM format
	ofstream fout( output_train_fname );
	for(int i=0;i<documents.size();i++){
		fout << labels[i] << " ";
		Document doc = documents[i];
		SparseVec phi = feaVect(doc);
		for(SparseVec::iterator it=phi.begin(); it!=phi.end(); it++)
			fout << it->first+1 << ":" << it->second << " ";
		fout << endl;
	}
	fout.close();

	// convert test docs to SVM format
	fout.open( output_test_fname );
	for(int i=0;i<test_documents.size();i++){
		fout << test_labels[i] << " ";
		Document doc = test_documents[i];
		SparseVec phi = feaVect(doc);
		for(SparseVec::iterator it=phi.begin(); it!=phi.end(); it++)
			fout << it->first+1 << ":" << it->second << " ";
		fout << endl;
	}
	fout.close();
	
}
