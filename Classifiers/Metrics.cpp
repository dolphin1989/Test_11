#include "Metrics.h"

float Metrics::computeEuc(const float* vec1, const float* vec2, const int dim)
{
	float res = 0.0;
	int i;
	for (i = 0; i < dim; i++)
	{
		res = res + (vec1[i] - vec2[i])*(vec1[i] - vec2[i]);
	}

	return res;
}

float Metrics::computeCos(const float* vec1, const float* vec2, const int dim)
{
	float res = 0.0, d1 = 0.0, d2 = 0.0;
	int i;
	for (i = 0; i < dim; i++)
	{
		res = res + vec1[i] * vec2[i];
	}
	for (i = 0; i < dim; i++)
	{
		d1 = d1 + vec1[i] * vec1[i];
	}
	d1 = sqrt(d1);
	for (i = 0; i < dim; i++)
	{
		d2 = d2 + vec2[i] * vec2[i];
	}
	d2 = sqrt(d2);
	assert(d1);
	assert(d2);

	return res/(d1*d2);
}

float Metrics::computeMah(const float* vec1, const float* vec2, const float* const* MMatrix, const int dim)
{
	float res = 0.0;
	int i, j;
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			res += vec1[i] * MMatrix[i][j] * vec2[j];
		}
	}

	return res;
}