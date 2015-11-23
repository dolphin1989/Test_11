#include "MQDF.h"
#include "Metrics.h"

MQDF::MQDF(int classnum, int dim, int kk)
{
	cnum = classnum;
	cdim = dim;
	knum = kk;

	pi = new float[cnum];
	memset(pi, 0, sizeof(float)*cnum);
	mean = new float[cnum*cdim];
	memset(mean, 0, sizeof(float)*cnum*cdim);

	vars = new float**[cnum];
	for (int i = 0; i < cnum; i++)
	{
		vars[i] = mat.allocMat(cdim);
	}
	for (int i = 0; i < cnum; i++)
	{
		for (int j = 0; j < cdim; j++)
		{
			for (int k = 0; k < cdim; k++)
			{
				vars[i][j][k] = 0;
			}
		}
	}

	eigval = new float[cnum*cdim];
	memset(eigval, 0, sizeof(float)*cnum*cdim);
	eigvec = new float*[cnum];
	for (int i = 0; i < cnum; i++)
	{
		eigvec[i] = new float[cdim*cdim];
		memset(eigvec[i], 0, sizeof(float)*cdim*cdim);
	}

	for (int i = 0; i < cnum; i++)
	{
		for (int j = 0; j < cdim*cdim; j++)
		{
			eigvec[i][j] = 0;
		}
	}
}

MQDF::~MQDF()
{
	if (pi)
		delete[] pi;
	if (mean)
		delete[] mean;
	if (vars)
	{
		for (int i = 0; i < cnum; i++)
		{
			mat.freeMat(vars[i], cdim);
		}

		delete[] vars;
	}
	if (eigval)
		delete[] eigval;
	if (eigvec)
	{
		for (int i = 0; i < cnum; i++)
		{
			delete[] eigvec[i];
		}
		delete[] eigvec;
	}
}

void MQDF::UpdateMeans(float* data, int* labels, int* ccou, int num)
{
	int i, j, tmp;
	for (i = 0; i < num; i++)
	{
		tmp = labels[i];
		if (tmp < 0)
		{
			continue;
		}

		assert(ccou[tmp] != 0);
		for (j = 0; j < cdim; j++)
		{
			mean[tmp*cdim + j] += (1.0 / ccou[tmp])*data[i*cdim + j];
		}
	}
}

void MQDF::UpdateVars(float* data, int* labels, int* ccou, int num)
{
	int i, j, k, tmp;
	float** diff;
	bool isSingleSample = false;
	diff = mat.allocMat(cdim);

	for (i = 0; i < cnum; i++)
	{
		if (ccou[i] == 1)
		{
			isSingleSample = true;
			break;
		}
	}

	for (i = 0; i < num; i++)
	{
		tmp = labels[i];
		if (tmp < 0)
			continue;

		assert(ccou[tmp] != 0);
		for (j = 0; j < cdim; j++)
		{
			for (k = 0; k < cdim; k++)
			{
				diff[j][k] = (data[i*cdim + j] - mean[tmp*cdim + j])*(data[i*cdim + k] - mean[tmp*cdim + k]);
			}
		}

		assert(ccou[tmp] != 0);
		if (isSingleSample){
			for (j = 0; j < cdim; j++){
				for (k = 0; k < cdim; k++)
				{
					vars[tmp][j][k] += (1.0 / ccou[tmp])*diff[j][k];
				}
			}
		}
		else
		{
			for (j = 0; j < cdim; j++){
				for (k = 0; k < cdim; k++)
				{
					vars[tmp][j][k] += (1.0 / (ccou[tmp] - 1))*diff[j][k];
				}
			}
		}
	}

	mat.freeMat(diff, cdim);
}

void MQDF::UpdateEigs()
{
	int i, j, k;
	float** A, **P;
	A = mat.allocMat(cdim);
	P = mat.allocMat(cdim);

	for (i = 0; i < cnum; i++)
	{
		mat.KLT(vars[i], cdim, eigvec[i], eigval+i*cdim);
	}
}

void MQDF::ComputeDis(const float* data, float* dis1, float* dis2)
{
	int i, j;
	float* tmp;
	float tmpDis, tmppartInner;
	tmp = new float[cdim];
	for (i = 0; i < cnum; i++)
	{
		for (j = 0; j < cdim; j++)
		{
			tmp[j] = data[j] - mean[i*cdim + j];
		}

		tmpDis = mat.innProduct(tmp, tmp, cdim);
		dis2[i] = tmpDis;
		for (j = 0; j < knum; j++)
		{
			tmppartInner = mat.innProduct(tmp, eigvec[i] + j*cdim, cdim);
			dis1[i] += (1.0 / eigval[i*cdim + j]) * tmppartInner*tmppartInner;
			dis2[i] -= tmppartInner*tmppartInner;
		}
	}
}

void MQDF::Train(float* traindata, int* labels, int num)
{
	int i;
	int* ccou;
	ccou = new int[cnum];
	memset(ccou, 0, sizeof(int)*cnum);

	for (i = 0; i < num; i++)
	{
		ccou[labels[i]]++;
	}
	for (i = 0; i < cnum; i++)
	{
		pi[i] = (1.0*ccou[i]) / num;
	}

	UpdateMeans(traindata, labels, ccou, num);
	UpdateVars(traindata, labels, ccou, num);
	UpdateEigs();
}

int MQDF::classify(const float* data)
{
	int label = -1;
	float* dis1;
	float* dis2;
	float posVal, tmpVal;
	int i, j;

	dis1 = new float[cnum];
	dis2 = new float[cnum];
	memset(dis1, 0, sizeof(float)*cnum);
	memset(dis2, 0, sizeof(float)*cnum);
	ComputeDis(data, dis1, dis2);
	for (i = 0; i < cnum; i++)
	{
		tmpVal = -1 * dis1[i] - (1.0 / eigval[i*cdim + knum - 1])*dis2[i] - (cdim - knum)*log(eigval[i*cdim + knum - 1]);
		for (j = 0; j < knum; j++)
		{
			tmpVal -= log(eigval[i*cdim + j]);
		}

		if (0 == i)
		{
			posVal = tmpVal;
			label = 0;
		}
		else if (tmpVal > posVal)
		{
			posVal = tmpVal;
			label = i;
		}
	}

	if (dis1)
		delete[] dis1;
	if (dis2)
		delete[] dis2;
	return label;
}