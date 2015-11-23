#include "QDF.h"


QDF::QDF(int classnum, int dim)
{
	cnum = classnum;
	cdim = dim;

	mean = new float[cnum*cdim];
	memset(mean, 0, sizeof(float)*cnum*cdim);

	vars = new float**[cnum];
	for (int i = 0; i < cnum; i++)
	{
		vars[i] = mat.allocMat(cdim);
	}
	varsInv = new float**[cnum];
	for (int i = 0; i < cnum; i++)
	{
		varsInv[i] = mat.allocMat(cdim);
	}
	varsldet = new float[cnum];

	for (int i = 0; i < cnum; i++){
		for (int j = 0; j < cdim; j++){
			for (int k = 0; k < cdim; k++)
			{
				vars[i][j][k] = 0;
			}
		}
	}

	pi = new float[cnum];
	memset(pi, 0, sizeof(float)*cnum);
}

QDF::~QDF()
{
	if (mean)
		delete[] mean;
	if (vars)
	{
		for (int i = 0; i < cnum; i++)
			mat.freeMat(vars[i], cdim);

		delete[] vars;
	}
	if (varsInv)
	{
		for (int i = 0; i < cnum; i++)
			mat.freeMat(varsInv[i], cdim);

		delete[] varsInv;
	}
	if (varsldet)
		delete[] varsldet;
	if (pi)
		delete[] pi;
}

void QDF::UpdateMeans(float* data, int* labels, int* ccou, int num)
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

void QDF::UpdateVars(float* data, int* labels, int* ccou, int num)
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

	for (i = 0; i < cnum; i++)
	{
		varsldet[i] = mat.matInverse(vars[i], cdim, varsInv[i]);
	}
}

void QDF::Train(float* traindata, int* labels, int num)
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
}

int QDF::classify(float* data)
{
	int label = -1;
	float* tmp;
	float posVal, tmpVal;
	int i, j;

	tmp = new float[cdim];
	memset(tmp, 0, sizeof(float)*cdim);

	for (i = 0; i < cnum; i++)
	{
		for (j = 0; j < cdim; j++)
		{
			tmp[j] = data[j] - mean[i*cdim + j];
		}

		tmpVal = -1 * Metrics::computeMah(tmp, tmp, varsInv[i], cdim) - varsldet[i];
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
	if (tmp)
		delete[] tmp;

	return label;
}
