#include "FileOpr.h"

FileOpr::FileOpr(const char* filename)
{
	int i, j;
	char tmp;
	fstream *in = new fstream(filename);
	if (!in->is_open())
	{
		cout << filename << " can not be read!";
		exit(-1);
	}

	*in >> datanum >> dim;
	assert(datanum != 0 && dim != 0);

	data = new trType[datanum*dim];
	memset(data, 0, sizeof(trType)*datanum*dim);
	classlabel = new char[datanum];
	memset(classlabel, 0, sizeof(char)*datanum);

	for (i = 0; i < datanum; i++)
	{
		*in >> classlabel[i];
		for (j = 0; j < dim; j++)
		{
			*in >> tmp;
			*in >> data[i*dim + j];
		}
	}

	assert(in != NULL);
	in->close();
}

FileOpr::~FileOpr()
{
	if (data)
	{
		delete[] data;
	}

	if (classlabel)
	{
		delete[] classlabel;
	}
}

int FileOpr::GetDim()
{
	return dim;
}

int FileOpr::GetNum()
{
	return datanum;
}

int FileOpr::GetClassNum()
{
	int res = 0;
	int i, j;
	assert(classlabel != NULL);
	for (i = 0; i < datanum; i++)
	{
		for (j = 0; j < i; j++)
		{
			if (classlabel[j] == classlabel[i])
			{
				break;
			}
		}

		if (i == j)
		{
			res++;
		}
	}

	return res;
}

void FileOpr::SplitData(trType* traindata, char* trainlbl, trType* tstdata, char *tstlbl, int trainnum)
{
	int tstnum;
	int i, j;

	tstnum = datanum - trainnum;
	///TransformLabelToInt(classlabel, datanum);

	for (i = 0; i < trainnum; i++)
	{
		trainlbl[i] = classlabel[i];
		for (j = 0; j < dim; j++)
		{
			traindata[i*dim + j] = data[i*dim + j];
		}
	}

	for (i = 0; i < tstnum; i++)
	{
		tstlbl[i] = classlabel[trainnum + i];
		for (j = 0; j < dim; j++)
		{
			tstdata[i*dim + j] = data[(i + trainnum)*dim + j];
		}
	}
}

void FileOpr::test()
{
	FileOpr f("C:\\Users\\Administrator\\Documents\\Visual Studio 2013\\Projects\\LVQ_LOGM\\LVQ_LOGM\\Data\\letter.data");
	
	int i, j;
	int trainnum = 16000;
	int dim = f.GetDim();
	int datanum = f.GetNum();
	int testnum = datanum - trainnum;

	trType* traindata = new trType[trainnum*dim];
	memset(traindata, 0, sizeof(trType)*trainnum*dim);
	trType* testdata = new trType[testnum*dim];
	memset(testdata, 0, sizeof(trType)*testnum*dim);
	char* trainlbl = new char[trainnum];
	char* tstlbl = new char[testnum];
	memset(trainlbl, 0, sizeof(char)*trainnum);
	memset(tstlbl, 0, sizeof(char)*testnum);

	f.SplitData(traindata, trainlbl, testdata, tstlbl, trainnum);

	cout << "*********Train Data************" << endl;
	cout << trainnum << "        " << dim;
	cout << endl;
	for (i = 0; i < trainnum; i++)
	{
		cout << trainlbl[i] << " ";
		for (j = 0; j < dim; j++)
		{
			cout << traindata[i*dim + j] << " ";
		}

		cout << endl;
	}

	cout << "********end***************" << endl << endl;
	
	cout << "************Test Data*************" << endl;
	cout << testnum << "       " << dim << endl;
	for (i = 0; i < testnum; i++)
	{
		cout << tstlbl[i] << " ";
		for (j = 0; j < dim; j++)
		{
			cout << testdata[i*dim + j] << " ";
		}

		cout << endl;
	}
	cout << "********end**************" << endl << endl;

	if (traindata)
	{
		delete[] traindata;
	}

	if (testdata)
	{
		delete[] testdata;
	}

	if (trainlbl)
	{
		delete[] trainlbl;
	}

	if (tstlbl)
	{
		delete[] tstlbl;
	}
}
