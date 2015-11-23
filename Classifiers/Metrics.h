#ifndef METRICS_H
#define METRICS_H
#include <iostream>
#include <assert.h>
using namespace std;
class Metrics{
public:
	static float computeEuc(const float*, const float*, const int);
	static float computeCos(const float*, const float*, const int);
	static float computeMah(const float*, const float*, const float* const*, const int);
};
#endif METRICS_H