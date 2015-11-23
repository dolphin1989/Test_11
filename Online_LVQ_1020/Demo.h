#ifndef DEMO_H
#define DEMO_H
#include <iostream>
#include <assert.h>
#include "datatype.h"

using namespace std;

class Demo{
public:
	void RunDemo(const char*);
	void TransfromLblToInt(const char *, int *, int);

	void test();
};

#endif