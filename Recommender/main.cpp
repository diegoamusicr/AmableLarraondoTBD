#include <iostream>
#include <numeric>
#include <time.h>
#include "omnislimtree.h"
#include "database.h"

extern void vecDiffWrapper(double *, double *, double *, int);
extern double reduceSumWrapper(double *, int);

#define ST_CAPACITY 25
#define ST_FOCI 10

typedef vector<double> movie;

using namespace std;

template <class T>
void printVector(vector<T> a)
{
	for (auto i = a.begin(); i != a.end(); i++)
		cout << *i << " ";
	cout << endl;
}

vector<double> genVector(int n)
{
	vector<double> v(n);
	for (int i=0; i < n; i++)
		v[i] = ((double) rand()/RAND_MAX);
	return v;
}

CSVReader * database;

double ManhattanDist(vector<double> &a, vector<double> &b)
{
	double * A = a.data();
	double * B = b.data();
	double * C = new double[a.size()];

	vecDiffWrapper(A, B, C, a.size());

	double sum = reduceSumWrapper(C, a.size());

	delete C;

	return sum;
}

double MovieDist(int ID1, int ID2)
{
	return ManhattanDist(database->data->at(ID1), database->data->at(ID2));
}

class MovieRecommender
{
	CSVReader * R;
	SlimTree<movieID> * ST;
public:
	MovieRecommender(CSVReader * db)
	{
		R = db;
		ST = new SlimTree<movieID>(ST_CAPACITY, MovieDist);
	}

	movieID GetMovieID(int ID)
	{
		return R->dataID->at(ID);
	}

	void GenTree()
	{
		int cp_delta = 5;
		int cp_size = floor(R->data->size()*((double)cp_delta/100));
		vector<int> checkpoints((int)100/cp_delta);
		for (int i = 0; i < checkpoints.size(); i++)
		{
			checkpoints[i] = cp_size*(i+1); 
		}
		printVector(checkpoints);
		int cp = 0;
		for (int i = 0; i < R->data->size(); i++)
		{
			ST->AddElement(i);
			cout << i << endl;
			if (i == checkpoints[cp])
			{
				cp += 1;
				cout << "SlimTree " << cp*cp_delta << "\% filled! (" << i+1 << " movies)" << endl;
			}
		}
		ST->SlimDownLeaves();
		cout << "Slim-down done!" << endl;
		ST->FindFoci(ST_FOCI);
		cout << "Foci calculated!" << endl;
	}
	void Query(int ID)
	{
		double radius = 20.0;
		cout << "Movie ID: " << GetMovieID(ID) << endl;

		clock_t t = clock();

		vector<movieID> q = ST->RangeQuery(ID, radius);
		
		t = clock() - t;
		cout << "Range Query made in " << ((float)t)/CLOCKS_PER_SEC << " seconds." << endl;

		cout << "Results (radius of " << radius << ")" << endl;
		printVector(q);
	}
};

int main()
{
	/*SlimTree<int> * ST = new SlimTree<int>(5, dist);
	int a = 10;
	ST->AddElement(a);
	vector<int> Q = ST->RangeQuery(9, 2);
	srand(time(NULL));
	vector<double> A = genVector(1000);
	vector<double> B = genVector(1000);

	int test = 100;
	double diff;

	clock_t t = clock();
	for (int i=0; i < test; i++)
	{
		diff = ManhattanDist(A,B);
	}
	t = clock() - t;
	t = t / test;

	cout << diff << endl;
	cout << "Tiempo promedio: " << ((float)t)/CLOCKS_PER_SEC << endl;*/

	clock_t t = clock();
	database = new CSVReader("movie_database.csv", ",");
	database->getData();
	t = clock() - t;
	cout << "Database loaded in " << ((float)t)/CLOCKS_PER_SEC << " seconds." << endl;
	cout << "Database size: " << database->data->size() << " movies" << endl;
	
	t = clock();
	MovieRecommender MR(database);
	MR.GenTree();
	t = clock() - t;
	cout << "SlimTree loaded in " << ((float)t)/CLOCKS_PER_SEC << " seconds." << endl;

	MR.Query(100);

	return 0;
}