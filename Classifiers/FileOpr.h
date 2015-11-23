#ifndef FILEOPR_H
#define FILEOPR_H
#include <iostream>
#include <fstream>
#include <assert.h>
typedef float trType;

using namespace std;

class FileOpr{
private:
	int datanum;
	int dim;

	trType *data;
	char *classlabel;
	char transformStrToChar(char*);

public:
	FileOpr(const char*);
	FileOpr(){}
	~FileOpr();

	int GetDim();
	int GetNum();
	int GetClassNum();
	///void TransformLabelToInt(trType *, int);
	void SplitData(trType *, char *, trType *, char *, int, int*);
	void GetRandomArray(int*);
	void test();
};
#endif