#include "Parzen.h"
#include "Metrics.h"

Parzen::Parzen(int dim, int classnum, float* traindata, int* trainlabels, int datanum)
{
	cdim = dim;
	cnum = classnum;
	num = datanum;
	data = new float[cdim*num];
	memcpy(data, traindata, sizeof(float)*cdim*num);
	labels = new int[num];
	memcpy(labels, trainlabels, sizeof(int)*num);
	pw = new float[cnum];
	
	int* ccou = new int[cnum];
	memset(ccou, 0, sizeof(int)*cnum);
	for (int i = 0; i < num; i++)
	{
		ccou[trainlabels[i]]++;
	}
	for (int i = 0; i < cnum; i++)
	{
		pw[i] = (1.0*ccou[i]) / num;
	}
	if (ccou)
		delete[] ccou;
}

Parzen::~Parzen()
{
	if (data)
		delete[] data;
	if (labels)
		delete[] labels;
	if (pw)
		delete[] pw;
}

int Parzen::classify(float* testdata)
{
	float maxtmp, tmp;
	int i, j, k, label = -1;
	for (i = 0; i < cnum; i++)
	{
		tmp = 0;
		for (j = 0; j < num; j++)
		{
			if (labels[j] != i)
				continue;
			tmp += (1.0 / pow(2 * PI*h*h, cdim / 2.0))*exp(-1 * (Metrics::computeEuc(testdata, data+j*cdim, cdim)) / (2.0*h*h));
		}
		tmp = tmp / num;
		if (i == 0)
		{
			maxtmp = tmp;
			label = i;
		}
		else
		{
			if (tmp > maxtmp)
			{
				maxtmp = tmp;
				label = i;
			}
		}
	}

	return label;
}

float Parzen::Test(float* traindata, int* trainlabel, int trainnum, float* testdata, int* testlabel, int testnum, float hh)
{
	float maxtmp, tmp;
	int i, j, k, l, label = -1, err = 0;
	for (i = 0; i < testnum; i++)
	{
		for (j = 0; j < cnum; j++)
		{
			tmp = 0;
			for (k = 0; k < trainnum; k++)
			{
				if (trainlabel[k] != j)
					continue;
	
				///tmp += (1.0 / pow(2 * PI*hh*hh, cdim / 2.0))*exp(-1 * (Metrics::computeEuc(tmpdata, tmpdata, cdim)) / (2.0*hh*hh));
				tmp += (1.0 / pow(2 * PI*hh*hh, cdim / 2.0))*exp(-1 * (Metrics::computeEuc(testdata + i*cdim, traindata + k*cdim, cdim)) / (2.0*hh*hh));
			}
			tmp = tmp / trainnum;

			if (j == 0)
			{
				maxtmp = tmp;
				label = j;
			}
			else
			{
				if (tmp > maxtmp)
				{
					maxtmp = tmp;
					label = j;
				}
			}
		}

		if (label != testlabel[i])
		{
			err++;
		}
	}

	return (1.0*err) / testnum;
}

void Parzen::kCV(float* hs, int h_num, int k)
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
	for (m = 0; m < h_num; m++){
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
						tsdata[tsind*cdim + l] = data[j*cdim + l];
					}
					tslabel[tsind] = labels[j];
					tsind++;
				}
				else
				{
					for (l = 0; l < cdim; l++)
					{
						ttdata[ttind*cdim + l] = data[j*cdim + l];
					}
					ttlabel[ttind] = labels[j];
					ttind++;
				}
			}

			if (i == 0)
			{
				tmp_rate = Test(ttdata, ttlabel, ttnum, tsdata, tslabel, tsnum, hs[m]);
			}
			else
			{
				tmp_rate += Test(ttdata, ttlabel, ttnum, tsdata, tslabel, tsnum, hs[m]);
			}
		}
		tmp_rate = tmp_rate / k;

		if (m == 0)
		{
			err_rate = tmp_rate;
			h = hs[m];
		}
		else
		{
			if (tmp_rate < err_rate)
			{
				err_rate = tmp_rate;
				h = hs[m];
			}
		}
	}

	cout << "Parzen h: " << h << endl;
}

