#ifndef RDA_H
#define RDA_H
#include <iostream>
#include <assert.h>
#include "Matrix.h"
#include "Metrics.h"

using namespace std;

class RDA{
private:
	int cnum;
	int cdim;
	float* pi;
	float* mean;
	float*** vars;
	float** varcommon;
	float* sigma;
	float*** varsInv;
	float* varsldet;
	CMatrix mat;
	void UpdateMeans(float*, int*, int*, int);
	void UpdateVars(float*, int*, int*, int, float, float);
	float Test(float* , int* , int);

public:
	RDA(int, int);
	~RDA();
	void kCV(float*, int*, int, int, float*, float*, int, int, float&, float&);
	int classify(float*);
	void Train(float*, int*, int, float, float);
};
#endif