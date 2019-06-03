#include <iostream>
#include <string>
#include <fstream>
#include <boost/algorithm/string.hpp>

using namespace std;

typedef double dataGen;
typedef int movieID;

class CSVReader
{
	bool header;
	string fileName;
	string delimeter;

public:
	vector<vector<dataGen> > * data;
	vector<movieID> * dataID;

	CSVReader(string filename, string delm = ",", bool h = 1)
	{
		fileName = filename;
		delimeter = delm;
		header = h;
		data = new vector<vector<dataGen> >();
		dataID = new vector<int>();
	}
 	void getData();
};

void CSVReader::getData()
{
	data->clear();
	dataID->clear();

	ifstream file(fileName);
 
	string line = "";

	if (header)	getline(file, line);

	while (getline(file, line))
	{
		vector<dataGen> dataVec;
		vector<string> tmp;
		boost::algorithm::split(tmp, line, boost::is_any_of(delimeter));
		for (int i=0; i < tmp.size(); i++)
		{
			if (i == 0) dataID->push_back(stoi(tmp[i]));
			else dataVec.push_back(stod(tmp[i]));
		}
		data->push_back(dataVec);
	}

	file.close();
}