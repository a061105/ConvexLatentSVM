#include <map>
#include <string>
#include <iostream>
using namespace std;

int main(){
	
	string str = "Sung-En";
	
	map<string, int> id_map;

	//id_map[ str ] = 45;
	
	cout << (id_map.find(str)==id_map.end()) << endl;
}
