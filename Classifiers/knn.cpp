#include "knn.h"
#include "Metrics.h"

KNN::KNN(int dim, int classnum, float* traindata, int* trainlabels, int datanum)
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

KNN::~KNN()
{
	if (data)
		delete[] data;
	if (labels)
		delete[] labels;
	if (pw)
		delete[] pw;
}

int KNN::classify(float* testdata)
{
	int *res, *b;
	float* dis;
	int i, j, l, label, m = -1, err = 0;
	float tmp;
	res = new int[k];
	memset(res, -1, sizeof(int)*k);
	dis = new float[k];
	memset(dis, -1, sizeof(float)*k);
	b = new int[cnum];
	memset(b, 0, sizeof(int)*cnum);

	for (i = 0; i < cnum; i++)
	{
		tmp = 0;
		for (j = 0; j < num; j++)
		{
			tmp = Metrics::computeCos(testdata, data + j*cdim, cdim);
			Getknn(tmp, labels[j], j, res, dis);
		}

		for (j = 0; j < k; j++)
		{
			b[res[j]]++;
		}
		for (j = 0; j < cnum; j++)
		{
			if (b[j]>m)
			{
				m = b[j];
				label = j;
			}
		}
	}

	if (res)
		delete[] res;
	if (dis)
		delete[] dis;
	if (b)
		delete[] b;
	return label;
}

void KNN::Getknn(float tmp, int label, int kj, int* kindex, float* kdis)
{
	int i, j;
	if (kj < k)
	{
		kindex[kj] = label;
		kdis[kj] = tmp;

		if (kj != 0)
		{
			for (i = 0; i < kj; i++)
			{
				if (tmp > kdis[i])
					break;
			}
			if (i < kj)
			{
				for (j = kj - 1; j >= i; j--)
				{
					kdis[j + 1] = kdis[j];
					kindex[j + 1] = kindex[j];
				}
				kdis[i] = tmp;
				kindex[i] = label;
			}
		}
	}
	else
	{
		for (i = 0; i < k; i++)
		{
			if (tmp>kdis[i])
				break;
		}
		if (i < k)
		{
			for (j = k - 2; j>=i; j--)
			{
				kdis[j + 1] = kdis[j];
				kindex[j + 1] = kindex[j];
			}
			kdis[i] = tmp;
			kindex[i] = label;
		}
	}
}

float KNN::Test(float* traindata, int* trainlabel, int trainnum, float* testdata, int* testlabel, int testnum)
{
	int *res, *b;
	float *dis;
	int i, j, l, label, m = -1, err = 0;
	float tmp;
	res = new int[k];
	memset(res, -1, sizeof(int)*k);
	dis = new float[k];
	memset(dis, -1, sizeof(float)*k);
	b = new int[cnum];
	memset(b, 0, sizeof(int)*cnum);

	for (i = 0; i < testnum; i++)
	{
		for (j = 0; j < trainnum; j++)
		{
			tmp = Metrics::computeCos(testdata + i*cdim, traindata + j*cdim, cdim);
			Getknn(tmp, trainlabel[j], j, res, dis);
		}
		
		for (j = 0; j < k; j++)
		{
			b[res[j]]++;
		}
		for (j = 0; j < cnum; j++)
		{
			if (b[j]>m)
			{
				m = b[j];
				label = j;
			}
		}

		if (label != testlabel[i])
		{
			err++;
		}
	}

	if (res)
		delete[] res;
	if (dis)
		delete[] dis;
	if (b)
		delete[] b;

	return (1.0*err) / testnum;
}

void KNN::kCV(int* ks, int k_num, int kc)
{
	int everynum = num / kc;
	int pre_k;
	int i, j, l;
	int m, n;
	int ttind, tsind;
	int* kfold;
	int ttnum = everynum*(kc - 1);
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
	for (m = 0; m < k_num; m++){
		tmp_rate = 0;
		k = ks[m];

		for (i = 0; i < kc; i++)
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
				tmp_rate = Test(ttdata, ttlabel, ttnum, tsdata, tslabel, tsnum);
			}
			else
			{
				tmp_rate += Test(ttdata, ttlabel, ttnum, tsdata, tslabel, tsnum);
			}
		}
		tmp_rate = tmp_rate / kc;

		if (m == 0)
		{
			err_rate = tmp_rate;
			pre_k = ks[m];
		}
		else
		{
			if (tmp_rate < err_rate)
			{
				err_rate = tmp_rate;
				pre_k = ks[m];
			}
		}

		k = pre_k;
	}

	cout << "knn k: " << k << endl;
}

