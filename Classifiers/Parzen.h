#ifndef PARZEN_H
#define PARZEN_H
#include <iostream>
using namespace std;
#define PI 3.1415926
class Parzen{
private:
	int cnum;
	int cdim;
	int num;
	float h;
	float* data;
	int* labels;
	float* pw;
	float Test(float*, int*, int, float*, int*, int, float);
public:
	Parzen(int, int, float*, int*, int);
	~Parzen();
	void kCV(float* , int , int);
	int classify(float* );
};
#endif