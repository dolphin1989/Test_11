#include "Demo.h"
#include "FileOpr.h"
#include "LDF.h"
#include "QDF.h"
#include "MQDF.h"
#include "RDA.h"
#include "Parzen.h"
#include "knn.h"

void Demo::RunDemo(const char* filename)
{
	///Read train data and test data
	FileOpr f(filename);
	int i, j = 0;
	int trainnum = 14000;
	int dim = f.GetDim();
	int datanum = f.GetNum();
	int cnum = f.GetClassNum();
	int testnum = datanum - trainnum;
	int* randArray = new int[datanum];

	float* traindata = new float[trainnum*dim];
	memset(traindata, 0, sizeof(float)*trainnum*dim);
	float* testdata = new float[testnum*dim];
	memset(testdata, 0, sizeof(float)*testnum*dim);
	char* trainlbl = new char[trainnum];
	char* tstlbl = new char[testnum];
	memset(trainlbl, 0, sizeof(char)*trainnum);
	memset(tstlbl, 0, sizeof(char)*testnum);
	int *traintruth = new int[trainnum];
	int *testtruth = new int[testnum];

	memset(traintruth, 0, sizeof(int)*trainnum);
	memset(testtruth, 0, sizeof(int)*testnum);

	//get a random array from 0 to datanum-1
	f.GetRandomArray(randArray);
	f.SplitData(traindata, trainlbl, testdata, tstlbl, trainnum, randArray);

	TransfromLblToInt(trainlbl, traintruth, trainnum);
	TransfromLblToInt(tstlbl, testtruth, testnum);
	int err;
	int tmp;

	LDF l(cnum, dim, 1, 1);
	l.Train(traindata, traintruth, trainnum);/**/
	err = 0;
	for (i = 0; i < testnum; i++)
	{
		tmp = l.classify(testdata + i*dim);
		if (testtruth[i] != tmp)
		{
			err++;
		}
	}

	cout << "The error rate of classification (LDF1) is: " << (1.0*err) / testnum << endl;

	LDF l1(cnum, dim, 2, NULL);
	l1.Train(traindata, traintruth, trainnum);/**/
	err = 0;
	for (i = 0; i < testnum; i++)
	{
		tmp = l1.classify(testdata + i*dim);
		if (testtruth[i] != tmp)
		{
			err++;
		}
	}

	cout << "The error rate of classification (LDF2) is: " << (1.0*err) / testnum << endl;

	QDF q(cnum, dim);
	q.Train(traindata, traintruth, trainnum);
	err = 0;
	for (i = 0; i < testnum; i++)
	{
		tmp = q.classify(testdata + i*dim);
		if (testtruth[i] != tmp)
		{
			err++;
		}
	}

	cout << "The error rate of classification (QDF) is: " << (1.0*err) / testnum << endl;

	/**/
	int knum = 16;
	MQDF m(cnum, dim, knum);
	m.Train(traindata, traintruth, trainnum);/**/
	err = 0;
	for (i = 0; i < testnum; i++)
	{
		tmp = m.classify(testdata + i*dim);
		if (testtruth[i] != tmp)
		{
			err++;
		}
	}

	cout << "The error rate of classification (MQDF) is: " << (1.0*err) / testnum << endl;


	/**/

	/**test for RDA**/
	const int a_num = 10, b_num = 10;
	float aph[a_num], beta[b_num];
	for (i = 0; i < 10; i++)
	{
		if (i == 0)
		{
			aph[i] = 0.1;
			beta[i] = 0.1;
		}
		else
		{
			aph[i] = aph[i - 1] * 0.1;
			beta[i] = beta[i - 1] * 0.1;
		}
	}
	RDA r(cnum, dim);
	float final_aph, final_beta;
	r.kCV(traindata, traintruth, trainnum, 10, aph, beta, a_num, b_num, final_aph, final_beta);
	cout << "final_aph: " << final_aph << " ;final_beta: " << final_beta << endl;
	r.Train(traindata, traintruth, trainnum, final_aph, final_beta);

	err = 0;
	for (i = 0; i < testnum; i++)
	{
		tmp = r.classify(testdata + i*dim);
		if (testtruth[i] != tmp)
		{
			err++;
		}
	}

	cout << "The error rate of classification (RDA) is: " << (1.0*err) / testnum << endl;/**/


	/**test for parzen**/
	Parzen p(dim, cnum, traindata, traintruth, trainnum);
	float hs[10] = { 0.5, 1, 1.5, 2, 2.5 };
	int h_num = 5;
	int k = 10;
	p.kCV(hs, h_num, k);
	err = 0;
	for (i = 0; i < testnum; i++)
	{
		tmp = p.classify(testdata + i*dim);
		if (testtruth[i] != tmp)
		{
			err++;
		}
	}

	cout << "The error rate of classification (PW) is: " << (1.0*err) / testnum << endl;/**/

	/** test for knn**/
	KNN kn(dim, cnum, traindata, traintruth, trainnum);
	int ks[5] = { 1, 3, 5, 7, 9 };
	int k_num = 5;
	int kj = 10;
	kn.kCV(ks, k_num, kj);
	err = 0;
	for (i = 0; i < testnum; i++)
	{
		tmp = kn.classify(testdata + i*dim);
		if (testtruth[i] != tmp)
		{
			err++;
		}
	}

	cout << "The error rate of classification (KNN) is: " << (1.0*err) / testnum << endl;/**/

	/**
	cout << "*******begin*******" << endl;
	for (i = 0; i < trainnum; i++)
	{
	cout << traintruth[i] << " ";
	}
	cout << endl;
	cout << "********end********" << endl;

	cout << "********begin*******" << endl;
	for (i = 0; i < testnum; i++)
	{
	cout << testtruth[i] << " ";
	}
	cout << "********end**********" << endl;
	**/

	if (randArray)
		delete[] randArray;

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

}

void Demo::TransfromLblToInt(const char *origalLbls, int* truth, int num)
{
	int i;
	for (i = 0; i < num; i++)
	{
		///for letter
		truth[i] = (int)(origalLbls[i] + '0' - 'A') - 48;
		///for wine
		///truth[i] = (int)(origalLbls[i]) - 49;
		/**for glass**
		if (origalLbls[i] <= '3')
		{
			truth[i] = (int)(origalLbls[i]) - 49;
		}
		else
		{
			truth[i] = (int)(origalLbls[i]) - 50;
		}/**/
		///for iris/pima-indians-diabetes
		///truth[i] = (int)(origalLbls[i]) - 48;

		assert(truth[i] >= 0);
	}
}

void Demo::Test()
{
	RunDemo("C:\\Users\\Administrator\\Documents\\Visual Studio 2013\\Projects\\Classifiers\\Classifiers\\Data\\letter.data");
}