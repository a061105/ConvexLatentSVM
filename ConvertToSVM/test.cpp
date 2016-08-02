#include <string>
#include <iostream>
#include <vector>
#include <map>
using namespace std;

void printVec( ostream& out, const vector<int>& v ){
	
	for(int i=0;i<v.size();i++)
		out << v[i] << " " ;
	out << endl;
}

int main(){
	
	vector<int> a;
	a.push_back(3);
	a.push_back(1);
	a.push_back(2);

	vector<int> b;
	b.push_back(3);
	b.push_back(1);
	b.push_back(2);

	
	vector<int> c;
	c.push_back(2);
	c.push_back(3);
	c.push_back(1);
	
	map<vector<int>, int> m;
	m.insert( make_pair( a, 1 ));
	m.insert( make_pair( b, 2 ));
	m.insert( make_pair( c, 3 ));

	for(map<vector<int>, int>::iterator it=m.begin(); it!=m.end(); it++){
		printVec( cout , it->first );
		cout << it->second << endl;
	}

	return 0;
}
