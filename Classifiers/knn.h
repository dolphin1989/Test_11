#ifndef KNN_H
#define KNN_H
#include <iostream>
using namespace std;

class KNN{
private:
	int cnum;
	int cdim;
	int num;
	int k;	//k-nn
	float* data;
	int* labels;
	float* pw;
	
	float Test(float*, int*, int, float*, int*, int);
public:
	KNN(int, int, float*, int*, int);
	~KNN();
	void kCV(int*, int, int);
	int classify(float*);
	void Getknn(float, int, int, int*, float*);
};
#endif