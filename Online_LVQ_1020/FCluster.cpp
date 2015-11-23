#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Cluster.h"

REAL CFCluster::cluster( REAL* data, int sampNum, int dim, REAL* center, 
						int cenNum, int* index )
{
	if( cenNum==1 )	// one cluster, mean of all samples
	{
		int n, i;
		for( i=0; i<dim; i++ )
			center[i] = 0;

		for( n=0; n<sampNum; n++ )
			for( i=0; i<dim; i++ )
				center[i] += data[(_int64)n*dim+i];

		for( i=0; i<dim; i++ )
			center[i] /= sampNum;

		REAL diff, vari;
		vari = 0;
		for( n=0; n<sampNum; n++ )
		{
			index[n] = 0;
			for( i=0; i<dim; i++ )
			{
				diff = (REAL)data[(_int64)n*dim+i]-center[i];
				vari += diff*diff;
			}
		}
		vari /= sampNum;	// variance

		index[0] = 0;
		return vari;
	}

	// initialization by random selection
	initial( data, sampNum, dim, center, cenNum );
	return FSCL( data, sampNum, dim, center, cenNum, 40 );

	// k-means clustering
	///return kmeans( data, sampNum, dim, center, cenNum, index, 100 );
}

// k-means data clustering
REAL CFCluster::kmeans( REAL* data, int sampNum, int dim, REAL* center, 
					   int cenNum, int* label, int iter )
{
	int *subnum;
	subnum = new int [cenNum];		// number of samples in a cluster

	for( int n=0; n<sampNum; n++ )
		label[n] = 0;

	int END;
	int index, m;
	int n, i;
	REAL dmin, total;

	for( int cycle=0; cycle<iter; cycle++ )	// at most 100 sweeps
	{
		END = 1;	// remains 1 if no label is changed
		total = 0;	// total square error
		for( n=0; n<sampNum; n++ )
		{
			index = nearest( data+(_int64)n*dim, dim, center, cenNum, dmin );
			total += dmin;
			if( index!=label[n] )	// cluster label is changed
			{
				END = 0;
				label[n] = index;
			}
		}
		total /= sampNum;	// average variance
		//printf( "%2d  %8.2f\n", cycle, total );

		if( END==1 )	// no label changed
			break;

		// update the cluster centers to be centers of attracted samples
		for( m=0; m<cenNum; m++ )
		{
			for( i=0; i<dim; i++ )
				center[m*dim+i] = 0;
			subnum[m] = 0;
		}
		for( n=0; n<sampNum; n++ )
		{
			for( i=0; i<dim; i++ )
				center[ label[n]*dim+i ] += data[(_int64)n*dim+i];
			subnum[ label[n] ] ++;
		}

		for( m=0; m<cenNum; m++ )
		{
			if( subnum[m]==0 )
				continue;
			for( i=0; i<dim; i++ )
				center[m*dim+i] /= subnum[m];
		}
	}

	delete subnum;

	return total;
}

// Initialize the prototypes to be randomly selected samples
void CFCluster::initial( REAL* data, int sampNum, int dim, REAL* center, int cenNum )
{
	int *di;	// index of orginal data for each cluster
	di = new int [cenNum];

	int m, k, i;
	char OK;
	for( m=0; m<cenNum; m++ )
	{
		do {
			di[m] = rand()%sampNum;
			// see if it coincides with any former prototype or not
			OK = 1;
			for( k=0; k<m; k++ )
			{
				if( di[m]==di[k] || equal(data+di[k]*dim, data+di[m]*dim, dim) )
				{
					OK = 0;
					break;
				}
			}
		} while( OK==0 );

		for( i=0; i<dim; i++ )
			center[m*dim+i] = data[ di[m]*dim+i ];
	}

	delete di;
}

// Two feature vectors equal or not
char CFCluster::equal( REAL* ftr1, REAL* ftr2, int dim )
{
	for( int i=0; i<dim; i++ )
	{
		if( ftr1[i] != ftr2[i] )
			return 0;
	}

	return 1;
}

// Search for the nearest cluster center
int CFCluster::nearest( REAL* input, int dim, REAL* center, int cenNum, REAL& dmin )
{
	REAL dist, diff;

	dmin = 0;
	int i;
	for( i=0; i<dim; i++ )
	{
		diff = (REAL)input[i]-center[i];
		dmin += diff*diff;
	}

	int index = 0;
	for( int m=1; m<cenNum; m++ )
	{
		dist = 0;
		for( i=0; i<dim; i++ )
		{
			diff = (REAL)input[i]-center[m*dim+i];
			dist += diff*diff;
			if( dist>=dmin )
				break;
		}

		if( dist<dmin )
		{
			dmin = dist;
			index = m;
		}
	}

	return index;
}

// Deterministic annealing
void CFCluster::DA( REAL* data, int sampNum, int dim, REAL* center, int cenNum, int iter )
{
	REAL T = 10000;
	REAL rate0 = (REAL)0.2;
	REAL rate;

	REAL *dist, *prob;
	dist = new REAL [cenNum];
	prob = new REAL [cenNum];

	int n, m;
	REAL sum, aved;
	REAL total;
	for( int cycle=0; cycle<iter; cycle++ )
	{
		rate = rate0*(iter-cycle)/iter;
		total = 0;
		for( n=0; n<sampNum; n++ )
		{
			distance( data+(_int64)n*dim, dim, center, cenNum, dist );

			sum = 0;
			for( m=0; m<cenNum; m++ )
			{
				prob[m] = (REAL)exp( -dist[m]/T );
				sum += prob[m];
			}
			for( m=0; m<cenNum; m++ )
				prob[m] /= sum;

			aved = 0;
			for( m=0; m<cenNum; m++ )
				aved += prob[m]*dist[m];
			total += aved;

			for( m=0; m<cenNum; m++ )
				modify( data+(_int64)n*dim, dim, center+m*dim, rate*prob[m] );
		}
		total /= sampNum;
		printf( "%2d  %8.2f\n", cycle, total );

		T *= (REAL)0.95;
	}

	delete dist;
	delete prob;
}

// WTA (winner-take-all) competitive learning
void CFCluster::CL( REAL* data, int sampNum, int dim, REAL* center, int cenNum, int iter )
{
	REAL rate0 = (REAL)0.2;
	REAL rate;

	REAL dmin, total;
	int index;

	int n;
	for( int cycle=0; cycle<iter; cycle++ )
	{
		rate = rate0*(iter-cycle)/iter;
		total = 0;
		for( n=0; n<sampNum; n++ )
		{
			index = nearest( data+(_int64)n*dim, dim, center, cenNum, dmin );
			total += dmin;

			modify( data+(_int64)n*dim, dim, center+index*dim, rate );
		}
		total /= sampNum;
		printf( "%2d  %8.2f\n", cycle, total );
	}
}



// Frequency-sensitive competitive learning
REAL CFCluster::FSCL( REAL* data, int sampNum, int dim, REAL* center, int cenNum, int iter )
{
	int* frequ;
	frequ = new int [cenNum];
	
	for( int m=0; m<cenNum; m++ )
		frequ[m] = 1;

	REAL rate0=(REAL)0.2;
	REAL rate;

	REAL dmin, total;
	int index;

	int n;
	for( int cycle=0; cycle<iter; cycle++ )
	{
		rate = rate0*(iter-cycle)/iter;
		total = 0;
		for( n=0; n<sampNum; n++ )
		{
			index = sensitive( data+(_int64)n*dim, dim, center, cenNum, frequ, dmin );
			frequ[index] ++;
			total += dmin;

			modify( data+(_int64)n*dim, dim, center+index*dim, rate );
		}
		total /= sampNum;
		//printf( "%2d  %8.2f\n", cycle, total );
	}

	if (frequ)
		delete[]frequ;
	return total;
}

// Center vetor updating
void CFCluster::modify( REAL* input, int dim, REAL* vect, REAL rate )
{
	for( int i=0; i<dim; i++ )
		vect[i] += rate*( (REAL)input[i]-vect[i] );
}

// Frequency-sensitive search
int CFCluster::sensitive( REAL* input, int dim, REAL* center, int cenNum, 
						int* frequ, REAL& dmin )
{
	int index;
	REAL fdist, diff;

	dmin = (REAL)1E20;

	int i;
	for( int m=0; m<cenNum; m++ )
	{
		fdist = 0;
		for( i=0; i<dim; i++ )
		{
			diff = center[m*dim+i]-input[i];
			fdist += diff*diff;
		}
		fdist *= sqrt( (REAL)frequ[m] );

		if( fdist<dmin )
		{
			index = m;
			dmin = fdist;
		}
	}
	dmin /= sqrt( (REAL)frequ[index] );

	return index;
}

// Distances to all cluster centers
void CFCluster::distance( REAL* input, int dim, REAL* center, int cenNum, REAL* dist )
{
	REAL diff;

	int i;
	for( int m=0; m<cenNum; m++ )
	{
		dist[m] = 0;
		for( i=0; i<dim; i++ )
		{
			diff = center[m*dim+i]-input[i];
			dist[m] += diff*diff;
		}
	}
}

// Retrieve the quntization error
REAL CFCluster::errComp( REAL* data, int sampNum, int dim, REAL* center, int cenNum )
{
	REAL total=0;
	REAL dmin;
	int index;

	for( int n=0; n<sampNum; n++ )
	{
		index = nearest( data+(_int64)n*dim, dim, center, cenNum, dmin );
		total += dmin;
	}
	total /= sampNum;

	return total;
}

void CFCluster::agglomerative( REAL* data, int sampNum, int dim, REAL* center, 
						int cenNum, int* index )
{
	int* frequ;
	frequ = new int [sampNum];

	REAL* distort;
	distort = new REAL [sampNum];

	REAL* proxim;
	proxim = new REAL [sampNum*(sampNum-1)/2];

	int m, n;
	for( n=0; n<sampNum; n++ )
	{
		index[n] = n;
		frequ[n] = 1;
		distort[n] = 0;
	}

	int pi = 0;
	for( m=0; m<sampNum; m++ )
		for( n=m+1; n<sampNum; n++ )
			//proxim[pi++] = E2dist( data+m*dim, data+(_int64)n*dim, dim );
			proxim[pi++] = fproxim( data+m*dim, data+(_int64)n*dim, dim, 1, 1 );

	int iteration, ii;
	iteration = sampNum-cenNum;

	REAL dmin;
	int mi, ni;
	for( ii=0; ii<iteration; ii++ )
	{
		dmin = (REAL)1E16;
		for( m=0; m<sampNum; m++ )
		{
			if( index[m]<m )	// already merged
				continue;
			for( n=m+1; n<sampNum; n++ )
			{
				if( index[n]<n )
					continue;

				pi = m*(2*sampNum-m-1)/2+n-m-1;
				//dincrem = proxim[pi]-distort[m]-distort[n];	// increment of distortion
				if( proxim[pi]<dmin )
				{
					dmin = proxim[pi];
					mi = m;
					ni = n;
				}
			}
		}

		cenMerge( data+mi*dim, data+ni*dim, dim, frequ[mi], frequ[ni] );
		index[ni] = mi;
		pi = mi*(2*sampNum-mi-1)/2+ni-mi-1;
		distort[mi] = proxim[pi];
		for( n=ni+1; n<sampNum; n++ )
		{
			if( index[n]==ni )
				index[n] = mi;
		}
		frequ[mi] += frequ[ni];
		frequ[ni] = 0;

		// Update the between-cluster distances
		for( m=0; m<sampNum; m++ )
		{
			if( index[m]<m )
				continue;
			
			if( m<mi )
			{
				pi = m*(2*sampNum-m-1)/2+mi-m-1;
				//proxim[pi] = E2dist( data+m*dim, data+mi*dim, dim );
				//proxim[pi] *= frequ[m]+frequ[mi];
				proxim[pi] = fproxim( data+m*dim, data+mi*dim, dim, frequ[m], frequ[mi] );
			}
			else if( m>mi )
			{
				pi = mi*(2*sampNum-mi-1)/2+m-mi-1;
				//proxim[pi] = E2dist( data+m*dim, data+mi*dim, dim );
				//proxim[pi] *= frequ[m]+frequ[mi];
				proxim[pi] = fproxim( data+m*dim, data+mi*dim, dim, frequ[m], frequ[mi] );
			}
		}
	}

	int ci = 0;
	for( n=0; n<sampNum; n++ )
	{
		if( index[n]==n )
		{
			memcpy( center+ci*dim, data+(_int64)n*dim, dim*sizeof(REAL) );
			index[n] = ci;
			ci ++;
		}
		else
			index[n] = index[ index[n] ];
	}

	/*REAL total = 0;
	for( n=0; n<sampNum; n++ )
		total += E2dist( data+(_int64)n*dim, center+index[n]*dim, dim );
	total /= sampNum;
	printf( "agglomerative: %8.2f\n", total );*/

	delete proxim;
	delete frequ;
	delete distort;
}

REAL CFCluster::fproxim( REAL* vec1, REAL* vec2, int dim, int f1, int f2 )
{
	REAL* tvec;
	tvec = new REAL [dim];
	memcpy( tvec, vec1, dim*sizeof(REAL) );

	cenMerge( tvec, vec2, dim, f1, f2 );

	REAL dist1, dist2;
	dist1 = E2dist( vec1, tvec, dim );
	dist2 = E2dist( vec2, tvec, dim );

	delete tvec;

	return f1*dist1+f2*dist2;
}

void CFCluster::cenMerge( REAL* vec1, REAL* vec2, int dim, int f1, int f2 )
{
	for( int i=0; i<dim; i++ )
		vec1[i] = (f1*vec1[i]+f2*vec2[i])/(f1+f2);
}

REAL CFCluster::E2dist( REAL* vec1, REAL* vec2, int dim )
{
	REAL diff, dist = 0;
	for( int i=0; i<dim; i++ )
	{
		diff = vec1[i]-vec2[i];
		dist += diff*diff;
	}

	return dist;
}
