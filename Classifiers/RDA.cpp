#include "RDA.h"

RDA::RDA(int classnum, int dim)
{
	cnum = classnum;
	cdim = dim;

	pi = new float[cnum];
	memset(pi, 0, sizeof(float)*cnum);
	mean = new float[cnum*cdim];

	vars = new float**[cnum];
	for (int i = 0; i < cnum; i++)
	{
		vars[i] = mat.allocMat(cdim);
	}
	varcommon = mat.allocMat(cdim);

	sigma = new float[cnum];
	varsInv = new float**[cnum];
	for (int i = 0; i < cnum; i++)
	{
		varsInv[i] = mat.allocMat(cdim);
	}
	varsldet = new float[cnum];
}

RDA::~RDA()
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
	if (varcommon)
		delete[] varcommon;
}

void RDA::UpdateMeans(float* data, int* labels, int* ccou, int num)
{
	int i, j, tmp;
	memset(mean, 0, sizeof(float)*cnum*cdim);
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

void RDA::UpdateVars(float* data, int* labels, int* ccou, int num, float aph, float beta)
{
	int i, j, k, tmp;
	float** diff;
	bool isSingleSample = false;
	diff = mat.allocMat(cdim);

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
	for (int i = 0; i < cdim; i++)
	for (int j = 0; j < cdim; j++)
	{
		varcommon[i][j] = 0;
	}
	memset(sigma, 0, sizeof(float)*cnum);

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
		for (j = 0; j < cdim; j++)
		{
			for (k = 0; k < cdim; k++)
			{
				varcommon[j][k] += (pi[i])*vars[i][j][k];
			}

			sigma[i] += vars[i][j][j] / cdim;
		}
	}

	for (i = 0; i < cnum; i++)
	{
		for (j = 0; j < cdim; j++)
		{
			for (k = 0; k < cdim; k++)
			{
				if (j == k)
				{
					vars[i][j][k] = (1 - aph)*((1 - beta)*vars[i][j][k] + beta*varcommon[j][k]) + aph*sigma[i];
				}
				else
				{
					vars[i][j][k] = (1 - aph)*((1 - beta)*vars[i][j][k] + beta*varcommon[j][k]);
				}
			}
		}
	}

	for (i = 0; i < cnum; i++)
	{
		varsldet[i] = mat.matInverse(vars[i], cdim, varsInv[i]);
	}
}

void RDA::Train(float* traindata, int* labels, int num, float aph, float beta)
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
	UpdateVars(traindata, labels, ccou, num, aph, beta);
}

int RDA::classify(float* data)
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

float RDA::Test(float* testdata, int* labels, int num)
{
	int i, tmp, err = 0;

	for (i = 0; i < num; i++)
	{
		tmp = classify(testdata + i*cdim);
		if (tmp != labels[i])
		{
			err++;
		}
	}

	return (1.0*err) / num;
}

void RDA::kCV(float* traindata, int* labels, int num, int k, float* aph, float* beta, int aph_num, int beta_num, float& final_aph, float& final_beta)
{
	int everynum = num / k;
	int i, j, l;
	int m, n;
	int ttind, tsind;
	int* kfold;
	int ttnum = everynum*(k - 1);
	int tsnum = everynum;
	float* ttdata = new float[ttnum*cdim];
	float* tsdata = new float[tsnum*cdim];
	int* ttlabel = new int[ttnum];
	int* tslabel = new int[tsnum];
	float err_rate, tmp_rate;

	kfold = new int[num];
	for (i = 0; i < num; i++)
	{
		kfold[i] = i / everynum;
	}
	for (m = 0; m < aph_num; m++){
		for (n = 0; n < beta_num; n++)
		{
			tmp_rate = 0;
			for (i = 0; i < k; i++)
			{
				tsind = 0;
				ttind = 0;

				for (j = 0; j < num; j++)
				{
					if (kfold[j] == i)
					{
						for (l = 0; l < cdim; l++)
						{
							tsdata[tsind*cdim + l] = traindata[j*cdim + l];
						}
						tslabel[tsind] = labels[j];
						tsind++;
					}
					else
					{
						for (l = 0; l < cdim; l++)
						{
							ttdata[ttind*cdim + l] = traindata[j*cdim + l];
						}
						ttlabel[ttind] = labels[j];
						ttind++;
					}
				}

				Train(ttdata, ttlabel, ttnum, aph[m], beta[n]);
				if (i == 0)
				{
					tmp_rate = Test(tsdata, tslabel, tsnum);
				}
				else
				{
					tmp_rate += Test(tsdata, tslabel, tsnum);
				}
			}
			tmp_rate = tmp_rate / k;
			
			if (m == 0 && n == 0)
			{
				err_rate = tmp_rate;
				final_aph = aph[m];
				final_beta = beta[n];
			}
			else
			{
				if (tmp_rate < err_rate)
				{
					err_rate = tmp_rate;
					final_aph = aph[m];
					final_beta = beta[n];
				}
			}
		}
	}
}

