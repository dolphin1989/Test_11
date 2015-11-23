#ifndef QDF_H
#define QDF_H
#include <iostream>
#include <assert.h>
#include "Matrix.h"
#include "Metrics.h"

using namespace std;
class QDF{
private:
	int cnum;
	int cdim;
	float* mean;
	float*** vars;
	float*** varsInv;
	float* varsldet;
	float* pi;
	CMatrix mat;
	void UpdateMeans(float*, int*, int*, int);
	void UpdateVars(float*, int*, int*, int);

public:
	QDF(int, int);
	~QDF();

	void Train(float*, int*, int);
	int classify(float*);
};

#endif QDF_H