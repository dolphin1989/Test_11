
///#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Cluster.h"

REAL CCluster::cluster(trType* data, int sampNum, int dim, REAL* center,
	int cenNum, int* index)
{
	if (cenNum == 1)	// one cluster, mean of all samples
	{
		int n, i;
		for (i = 0; i<dim; i++)
			center[i] = 0;

		for (n = 0; n<sampNum; n++)
		for (i = 0; i<dim; i++)
			center[i] += data[(_int64)n*dim + i];

		for (i = 0; i<dim; i++)
			center[i] /= sampNum;

		REAL diff, vari;
		vari = 0;
		for (n = 0; n<sampNum; n++)
		{
			index[n] = 0;
			for (i = 0; i<dim; i++)
			{
				diff = (REAL)data[(_int64)n*dim + i] - center[i];
				vari += diff*diff;
			}
		}
		vari /= (sampNum-1);	// variance

		return vari;
	}

	// initialization by random selection
	initial(data, sampNum, dim, center, cenNum);
	return FSCL(data, sampNum, dim, center, cenNum, 40);

	// k-means clustering
	////return kmeans(data, sampNum, dim, center, cenNum, index, 100);
}

// k-means data clustering
REAL CCluster::kmeans(trType* data, int sampNum, int dim, REAL* center,
	int cenNum, int* label, int iter)
{
	int *subnum;
	subnum = new int[cenNum];		// number of samples in a cluster

	for (int n = 0; n<sampNum; n++)
		label[n] = 0;

	int END;
	int index, m;
	int n, i;
	REAL dmin, total;

	for (int cycle = 0; cycle<iter; cycle++)	// at most 100 sweeps
	{
		END = 1;	// remains 1 if no label is changed
		total = 0;	// total square error
		for (n = 0; n<sampNum; n++)
		{
			index = nearest(data + (_int64)n*dim, dim, center, cenNum, dmin);
			total += dmin;
			if (index != label[n])	// cluster label is changed
			{
				END = 0;
				label[n] = index;
			}
		}
		total /= sampNum;	// average variance
		//printf( "%2d  %8.2f\n", cycle, total );

		if (END == 1)	// no label changed
			break;

		// update the cluster centers to be centers of attracted samples
		for (m = 0; m<cenNum; m++)
		{
			for (i = 0; i<dim; i++)
				center[m*dim + i] = 0;
			subnum[m] = 0;
		}
		for (n = 0; n<sampNum; n++)
		{
			for (i = 0; i<dim; i++)
				center[label[n] * dim + i] += data[(_int64)n*dim + i];
			subnum[label[n]] ++;
		}

		for (m = 0; m<cenNum; m++)
		{
			if (subnum[m] == 0)
				continue;
			for (i = 0; i<dim; i++)
				center[m*dim + i] /= subnum[m];
		}
	}

	delete subnum;

	return total;
}

// Initialize the prototypes to be randomly selected samples
void CCluster::initial(trType* data, int sampNum, int dim, REAL* center, int cenNum)
{
	int *di;	// index of orginal data for each cluster
	di = new int[cenNum];

	int m, k, i;
	char OK;
	for (m = 0; m<cenNum; m++)
	{
		do {
			di[m] = rand() % sampNum;
			// see if it coincides with any former prototype or not
			OK = 1;
			for (k = 0; k<m; k++)
			{
				if (di[m] == di[k] || equal(data + di[k] * dim, data + di[m] * dim, dim))
				{
					OK = 0;
					break;
				}
			}
		} while (OK == 0);

		for (i = 0; i<dim; i++)
			center[m*dim + i] = data[di[m] * dim + i];
	}

	delete di;
}

// Two feature vectors equal or not
char CCluster::equal(trType* ftr1, trType* ftr2, int dim)
{
	for (int i = 0; i<dim; i++)
	{
		if (ftr1[i] != ftr2[i])
			return 0;
	}

	return 1;
}

// Search for the nearest cluster center
int CCluster::nearest(trType* input, int dim, REAL* center, int cenNum, REAL& dmin)
{
	REAL dist, diff;

	dmin = 0;
	for (int i = 0; i<dim; i++)
	{
		diff = (REAL)input[i] - center[i];
		dmin += diff*diff;
	}

	int index = 0;
	int m, i;
	for (m = 1; m<cenNum; m++)
	{
		dist = 0;
		for (i = 0; i<dim; i++)
		{
			diff = (REAL)input[i] - center[m*dim + i];
			dist += diff*diff;
			if (dist >= dmin)
				break;
		}

		if (dist<dmin)
		{
			dmin = dist;
			index = m;
		}
	}

	return index;
}

// Deterministic annealing
void CCluster::DA(trType* data, int sampNum, int dim, REAL* center, int cenNum, int iter)
{
	REAL T = 10000;
	REAL rate0 = (REAL)0.2;
	REAL rate;

	REAL *dist, *prob;
	dist = new REAL[cenNum];
	prob = new REAL[cenNum];

	int n, m;
	REAL sum, aved;
	REAL total;
	for (int cycle = 0; cycle<iter; cycle++)
	{
		rate = rate0*(iter - cycle) / iter;
		total = 0;
		for (n = 0; n<sampNum; n++)
		{
			distance(data + (_int64)n*dim, dim, center, cenNum, dist);

			sum = 0;
			for (m = 0; m<cenNum; m++)
			{
				prob[m] = (REAL)exp(-dist[m] / T);
				sum += prob[m];
			}
			for (m = 0; m<cenNum; m++)
				prob[m] /= sum;

			aved = 0;
			for (m = 0; m<cenNum; m++)
				aved += prob[m] * dist[m];
			total += aved;

			for (m = 0; m<cenNum; m++)
				modify(data + (_int64)n*dim, dim, center + m*dim, rate*prob[m]);
		}
		total /= sampNum;
		printf("%2d  %8.2f\n", cycle, total);

		T *= (REAL)0.95;
	}

	delete dist;
	delete prob;
}

// WTA (winner-take-all) competitive learning
void CCluster::CL(trType* data, int sampNum, int dim, REAL* center, int cenNum, int iter)
{
	REAL rate0 = (REAL)0.2;
	REAL rate;

	REAL dmin, total;
	int index;

	int n;
	for (int cycle = 0; cycle<iter; cycle++)
	{
		rate = rate0*(iter - cycle) / iter;
		total = 0;
		for (n = 0; n<sampNum; n++)
		{
			index = nearest(data + (_int64)n*dim, dim, center, cenNum, dmin);
			total += dmin;

			modify(data + (_int64)n*dim, dim, center + index*dim, rate);
		}
		total /= sampNum;
		printf("%2d  %8.2f\n", cycle, total);
	}
}

// Frequency-sensitive competitive learning
REAL CCluster::FSCL(trType* data, int sampNum, int dim, REAL* center, int cenNum, int iter)
{
	int* frequ;
	frequ = new int[cenNum];

	for (int m = 0; m<cenNum; m++)
		frequ[m] = 1;

	REAL rate0 = (REAL)0.2;
	REAL rate;

	REAL dmin, total;
	int index;

	int n;
	for (int cycle = 0; cycle<iter; cycle++)
	{
		rate = rate0*(iter - cycle) / iter;
		total = 0;
		for (n = 0; n<sampNum; n++)
		{
			index = sensitive(data + (_int64)n*dim, dim, center, cenNum, frequ, dmin);
			frequ[index] ++;
			total += dmin;

			modify(data + (_int64)n*dim, dim, center + index*dim, rate);
		}
		total /= (sampNum-1);
		//printf( "%2d  %8.2f\n", cycle, total );
	}

	if (frequ)
	{
		delete[] frequ;
	}

	return total;
}

// Center vetor updating
void CCluster::modify(trType* input, int dim, REAL* vect, REAL rate)
{
	for (int i = 0; i<dim; i++)
		vect[i] += rate*((REAL)input[i] - vect[i]);
}

// Frequency-sensitive search
int CCluster::sensitive(trType* input, int dim, REAL* center, int cenNum,
	int* frequ, REAL& dmin)
{
	int index;
	REAL fdist, diff;

	dmin = (REAL)1E20;

	int i;
	for (int m = 0; m<cenNum; m++)
	{
		fdist = 0;
		for (i = 0; i<dim; i++)
		{
			diff = center[m*dim + i] - input[i];
			fdist += diff*diff;
		}
		fdist *= sqrt((REAL)frequ[m]);

		if (fdist<dmin)
		{
			index = m;
			dmin = fdist;
		}
	}
	dmin /= sqrt((REAL)frequ[index]);

	return index;
}

// Distances to all cluster centers
void CCluster::distance(trType* input, int dim, REAL* center, int cenNum, REAL* dist)
{
	REAL diff;

	int i;
	for (int m = 0; m<cenNum; m++)
	{
		dist[m] = 0;
		for (i = 0; i<dim; i++)
		{
			diff = center[m*dim + i] - input[i];
			dist[m] += diff*diff;
		}
	}
}

// Retrieve the quntization error
REAL CCluster::errComp(trType* data, int sampNum, int dim, REAL* center, int cenNum)
{
	REAL total = 0;
	REAL dmin;
	int index;

	for (int n = 0; n<sampNum; n++)
	{
		index = nearest(data + (_int64)n*dim, dim, center, cenNum, dmin);
		total += dmin;
	}
	total /= sampNum;

	return total;
}
