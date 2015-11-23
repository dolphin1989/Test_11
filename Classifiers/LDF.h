#ifndef LDF_H
#define LDF_H
#include <iostream>
#include <assert.h>
#include "Matrix.h"

using namespace std;
class LDF{
private:
	int cnum;
	int cdim;
	int ctype;
	float csigma;
	float* mean;
	float** var;
	float* pi;
	float* weights;
	float* constweights;
	CMatrix mat;
	void UpdateMeans(float*, int*, int*, int);
	void UpdateVars(float*, int*, int, int*, int);
	void UpdateWeights();

public:
	LDF(int, int, int, float);
	~LDF();

	void Train(float*, int*, int);
	int classify(float* );
};

#endif LDF_H