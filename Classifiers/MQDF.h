#ifndef MQDF_H
#define MQDF_H
#include <iostream>
#include <assert.h>
#include "Matrix.h"

using namespace std;
class MQDF{
private:
	int cnum;
	int cdim;
	int knum;
	float* pi;
	float* mean;
	float*** vars;
	float* eigval;
	float** eigvec;
	CMatrix mat;
	void ComputeDis(const float *, float *, float *);
	void UpdateMeans(float*, int*, int*, int);
	void UpdateVars(float*, int*, int*, int);
	void UpdateEigs();

public:
	MQDF(int, int, int);
	~MQDF();
	void Train(float*, int*, int);
	int classify(const float*);
};
#endif MQDF_H