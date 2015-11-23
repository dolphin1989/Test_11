#include "LDF.h"
#include "Metrics.h"

LDF::LDF(int classnum, int dim, int type, float sigma)
{
	cnum = classnum;
	cdim = dim;
	ctype = type;

	mean = new float[cnum*cdim];
	memset(mean, 0, sizeof(float)*cnum*cdim);

	var = mat.allocMat(cdim);

	for (int i = 0; i < cdim; i++){
		for (int j = 0; j < cdim; j++){
			{
				var[i][j] = 0;
			}
		}
	}

	pi = new float[cnum];
	memset(pi, 0, sizeof(float)*cnum);

	if (ctype == 1)
	{
		csigma = sigma;
	}
	else
	{
		weights = new float[cnum*cdim];
		memset(weights, 0, sizeof(float)*cnum*cdim);
		constweights = new float[cnum];
		memset(constweights, 0, sizeof(float)*cnum);
	}
}

LDF::~LDF()
{
	if (mean)
		delete[] mean;
	if (var)
	{
		mat.freeMat(var, cdim);
	}
	if (pi)
		delete[] pi;
	if (ctype != 1)
	{
		if (weights)
			delete[] weights;
		if (constweights)
			delete[] constweights;
	}
}

void LDF::UpdateMeans(float* data, int* labels, int* ccou, int num)
{
	int i, j, tmp;
	if (num == 0)
		return;

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

void LDF::UpdateVars(float* data, int* labels, int num, int* ccou, int type)
{
	int i;
	if (num == 0)
		return;

	if (type == 1)
	{
		for (i = 0; i < cdim; i++)
		{
			var[i][i] = csigma;
		}
	}
	else
	{
		int j, k, tmp;
		bool flag = false;
		float*** vars;
		float** diff;

		vars = new float**[cnum];
		for (i = 0; i < cnum; i++)
		{
			vars[i] = mat.allocMat(cdim);
			for (j = 0; j < cdim;j++)
			for (k = 0; k < cdim; k++)
			{
				vars[i][j][k] = 0;
			}
		}

		diff = mat.allocMat(cdim);

		for (i = 0; i < cnum; i++)
		{
			if (ccou[i] == 1)
			{
				flag = true;
				break;
			}
		}
		for (i = 0; i < num; i++)
		{
			tmp = labels[i];

			for (j = 0; j < cdim; j++)
			{
				for (k = 0; k < cdim; k++)
				{
					diff[j][k] = (data[i*cdim + j] - mean[tmp*cdim + j])*(data[i*cdim + k] - mean[tmp*cdim + k]);
				}
			}			
			
			if (flag)
			{
				for (j = 0; j < cdim; j++)
				{
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
		for (i = 0; i < cnum; i++)
		{
			for (j = 0; j < cdim; j++)
			{
				for (k = 0; k < cdim; k++)
				{
					var[j][k] += (pi[i])*vars[i][j][k];
				}
			}
		}

		if (vars)
		{
			for (i = 0; i < cnum; i++)
			{
				mat.freeMat(vars[i], cdim);
			}
		}
		if (diff)
		{
			mat.freeMat(diff, cdim);
		}
	}	
}

void LDF::Train(float* traindata, int* labels, int num)
{
	int i;
	int* ccou = new int[cnum];
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
	UpdateVars(traindata, labels, num, ccou, ctype);
	if (ctype != 1)
	{
		UpdateWeights();
	}
}

void LDF::UpdateWeights()
{
	int i, j, k;
	float** tmpInv = mat.allocMat(cdim);

	for (i = 0; i < cnum; i++)
	{
		mat.matInverse(var, cdim, tmpInv);
		for (j = 0; j < cdim; j++)
		{
			for (k = 0; k < cdim; k++)
			{
				weights[i*cdim + j] += 2*tmpInv[k][j] * mean[i*cdim + k];
			}
		}

		constweights[i] = 2 * log(pi[i]) - Metrics::computeMah(mean + i*cdim, mean + i*cdim, tmpInv, cdim);
	}

	mat.freeMat(tmpInv, cdim);
}

int LDF::classify(float* data)
{
	int label = -1, i;
	float obj, tmp;
	if (ctype == 1)
	{
		for (i = 0; i < cnum; i++)
		{
			tmp = Metrics::computeEuc(data, mean + i*cdim, cdim);
			tmp = -1 * tmp / csigma + 2*log(pi[i]);

			if (i != 0)
			{
				if (tmp>obj)
				{
					obj = tmp;
					label = i;
				}
			}
			else
			{
				obj = tmp;
				label = 0;
			}
		}
	}
	else
	{
		for (i = 0; i < cnum; i++)
		{
			tmp = mat.innProduct(weights + i*cdim, data, cdim) + constweights[i];

			if (i != 0)
			{
				if (tmp>obj)
				{
					obj = tmp;
					label = i;
				}
			}
			else
			{
				obj = tmp;
				label = 0;
			}
		}
	}

	return label;
}

