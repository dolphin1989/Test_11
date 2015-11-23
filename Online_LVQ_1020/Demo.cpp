#include "Demo.h"
#include "FileOpr.h"
#include "Classifier-LVQ.h"

void Demo::RunDemo(const char* filename)
{
	///Read train data and test data
	FileOpr f(filename);
	int i, j, k;
	int trainnum = 15000;
	int dim = f.GetDim();
	int datanum = f.GetNum();
	int cnum = f.GetClassNum();
	int testnum = datanum - trainnum;

	trType* traindata = new trType[trainnum*dim];
	memset(traindata, 0, sizeof(trType)*trainnum*dim);
	trType* testdata = new trType[testnum*dim];
	memset(testdata, 0, sizeof(trType)*testnum*dim);
	char* trainlbl = new char[trainnum];
	char* tstlbl = new char[testnum];
	memset(trainlbl, 0, sizeof(char)*trainnum);
	memset(tstlbl, 0, sizeof(char)*testnum);
	int *traintruth = new int[trainnum];
	int *testtruth = new int[testnum];
	memset(traintruth, 0, sizeof(int)*trainnum);
	memset(testtruth, 0, sizeof(int)*testnum);
	
	int firsttrainnum = 200;
	trType* firsttraindata = new trType[firsttrainnum*dim];
	trType* everytraindata = new trType[dim];

	int* firsttrainlbl = new int[firsttrainnum];
	int everytrainlbl = 0;

	f.SplitData(traindata, trainlbl, testdata, tstlbl, trainnum);
	TransfromLblToInt(trainlbl, traintruth, trainnum);
	TransfromLblToInt(tstlbl, testtruth, testnum);

	char *configr = new char[2];
	configr[0] = 'W';
	configr[1] = '2';
	int mpnum = 600;
	long totalnum;
	long totalIter;

	///int coar = 30;
	///CLVQ lvq(cnum, dim, configr, coar);
	CLVQ lvq(cnum, dim, mpnum, configr);

	memcpy(firsttraindata, traindata, sizeof(trType)*firsttrainnum*dim);
	memcpy(firsttrainlbl, traintruth, sizeof(int)*firsttrainnum);
	///lvq.AdjustProto(firsttraindata, firsttrainlbl, firsttrainnum, pronumforclasses);

	lvq.LVQtrain(firsttraindata, firsttrainnum, dim, firsttrainlbl, configr, 0.05, 40, 1, NULL);
	totalnum = firsttrainnum;
	totalIter = 1;
   
	REAL res = lvq.LVQtest(testdata, testnum, dim, cnum, testtruth, 0);
	cout << "The result is: " << res << endl;
	REAL thresh1 = 1;
	int counts = 1;
	REAL rate0 = 1;
	REAL regu = 0.5;
	REAL rate;

	int aa = trainnum - firsttrainnum;
	for (i = 0; i < aa; i++)
	{
		totalnum++;
		totalIter++;
		rate = rate0*exp(-1.0*totalIter / totalnum);
		k = i + firsttrainnum;
		for (j = 0; j < dim; j++)
		{
			everytraindata[j] = traindata[k*dim + j];
		}
		everytrainlbl = traintruth[k];

		lvq.OnlineTrain(everytraindata, everytrainlbl, configr, thresh1, counts, rate, regu);
	}

	res = lvq.LVQtest(testdata, testnum, dim, cnum, testtruth, 1);
	cout << "The result is: " << res << endl;

	///REAL regu0, int iteration, REAL relRate, int seed
	///CLVQ::CLVQ(int cnum, int dm, char* configr, int coar)

	if (firsttraindata)
		delete[] firsttraindata;
	if (firsttrainlbl)
		delete[] firsttrainlbl;
	if (everytraindata)
		delete[] everytraindata;

	if (traindata)
		delete[] traindata;
	if (testdata)
		delete[] testdata;
	if (trainlbl)
		delete[] trainlbl;
	if (tstlbl)
		delete[] tstlbl;
	if (traintruth)
		delete[] traintruth;
	if (testtruth)
		delete[] testtruth;

	if (configr)
		delete[] configr;
}

void Demo::TransfromLblToInt(const char *origalLbls, int* truth, int num)
{
	int i;
	for (i = 0; i < num; i++)
	{
		truth[i] = (int)(origalLbls[i] + '0' - 'A') - 48;
		assert(truth[i] >= 0);
	}
}

void Demo::test()
{
	RunDemo("C:\\Users\\Administrator\\Documents\\Visual Studio 2013\\Projects\\LVQ_LOGM\\LVQ_LOGM\\Data\\letter.data");
}