#include "FileOpr.h"
#include <time.h>

FileOpr::FileOpr(const char* filename)
{
	/**for letter/wine**/
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
	in->close();/**/
	/**for glass/pima-indians-diabetes**
	int i, j;
	char tmpchar;
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
		for (j = 0; j < dim; j++)
		{
			*in >> data[i*dim + j];
			*in >> tmpchar;
		}
		*in >> classlabel[i];
	}

	assert(in != NULL);
	in->close();/**/
	/**for iris**
	int i, j;
	char tmpstr[20];
	char tmpchar;
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
		for (j = 0; j < dim; j++)
		{
			*in >> data[i*dim + j];
			*in >> tmpchar;
		}
		*in >> tmpstr;
		classlabel[i] = transformStrToChar(tmpstr);
	}

	assert(in != NULL);
	in->close();/**/
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

char FileOpr::transformStrToChar(char* str)
{
	char res;
	if (!strcmp(str, "Iris-setosa"))
	{
		res = '0';
	}
	else if (!strcmp(str, "Iris-versicolor"))
	{
		res = '1';
	}
	else
	{
		res = '2';
	}

	return res;
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

void FileOpr::GetRandomArray(int *randomArray)
{
	int i, tmpi, tmpn;
	srand(time(NULL));

	for (i = 0; i < datanum; i++)
	{
		randomArray[i] = i;
	}
	for (i = 0; i < datanum; i++)
	{
		tmpi = rand() % datanum;
		tmpn = randomArray[tmpi];
		randomArray[tmpi] = randomArray[i];
		randomArray[i] = tmpn;
	}
}

void FileOpr::SplitData(trType* traindata, char* trainlbl, trType* tstdata, char *tstlbl, int trainnum, int* randArray)
{
	int tstnum;
	int i, j, k;

	tstnum = datanum - trainnum;

	for (i = 0; i < trainnum; i++)
	{
		k = randArray[i];
		trainlbl[i] = classlabel[k];
		for (j = 0; j < dim; j++)
		{
			traindata[i*dim + j] = data[k * dim + j];
		}
	}

	for (i = 0; i < tstnum; i++)
	{
		k = randArray[i + trainnum];
		tstlbl[i] = classlabel[k];
		for (j = 0; j < dim; j++)
		{
			tstdata[i*dim + j] = data[k*dim + j];
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

	//f.SplitData(traindata, trainlbl, testdata, tstlbl, trainnum);

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
