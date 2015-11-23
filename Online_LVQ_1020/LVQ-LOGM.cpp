
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Classifier-LVQ.h"

#define PLIMIT (REAL)1E-4

// Construction of CLVQ class for prototype training
CLVQ::CLVQ( int cnum, int dm, int mpnum, char* configr)
{
	classNum = cnum;
	dim = dm;
	maxpronum = mpnum;

	proNum = atoi( configr+1 );	// number of prototypes of each class
	cenNum = classNum*proNum;
	oncenNum = classNum*maxpronum;

	centers = new REAL [cenNum*dim];
	dthresh = new REAL [cenNum];
	//Add for online by yyshen
	prodthresh = new REAL[classNum*maxpronum];

	procenters = new REAL[oncenNum*dim];
	pronumforclasses = new int[classNum];
	memset(pronumforclasses, 0, sizeof(int)*classNum);
	frequencies = new int[oncenNum];
	memset(frequencies, 0, sizeof(int)*oncenNum);

	if( configr[0]=='W' || configr[0]=='w' )
	{
		WDist = 1;
		weights = new REAL [cenNum*dim];	// prototype dependent weights
		onweights = new REAL[oncenNum*dim]; // prototype dependent weights for online
	}
	else
	{
		WDist = 0;
		weights = NULL;
		onweights = NULL;
	}

	initial = 0;
}

// Load from a dictionary file
CLVQ::CLVQ( FILE* fp )
{
	fread( &classNum, sizeof(int), 1, fp );
	fread( &proNum, sizeof(int), 1, fp );
	fread( &dim, sizeof(int), 1, fp );
	fread( &WDist, sizeof(int), 1, fp );

	cenNum = classNum*proNum;	// total number of prototypes

	centers = new REAL [cenNum*dim];
	fread( centers, sizeof(REAL), cenNum*dim, fp );
	if( WDist )
	{
		weights = new REAL [cenNum*dim];
		fread( weights, sizeof(REAL), dim, fp );
	}
	else
		weights = NULL;

	dthresh = new REAL [cenNum];
	fread( dthresh, sizeof(REAL), cenNum, fp );

	fread( &variance, sizeof(REAL), 1, fp );
	printf( "variance: %6.1f\n", variance );

	fread( &coarNum, sizeof(int), 1, fp );
	if( coarNum )
	{
		printf( "coarse clusters: %d\n", coarNum );

		coarse = new REAL [coarNum*dim];
		fread( coarse, sizeof(REAL), coarNum*dim, fp );

		coarSize = new int [coarNum];
		fread( coarSize, sizeof(int), coarNum, fp );

		coarSet = new int* [coarNum];
		for( int k=0; k<coarNum; k++ )
		{
			coarSet[k] = new int [ coarSize[k] ];
			fread( coarSet[k], sizeof(int), coarSize[k], fp );
		}

		coarLabel = new int [cenNum];	// will be used in discriminative training
		REAL td;
		for( int m=0; m<cenNum; m++ )
			coarLabel[m] = REnearest( centers+m*dim, dim, coarse, coarNum, NULL, td );

		coarRank = 3*(coarNum*coarNum/cenNum+1);
		if( coarRank>coarNum-1 )
			coarRank = coarNum-1;
	}

	initial = 2;
}

CLVQ::~CLVQ()
{
	delete centers;
	delete dthresh;
	if (WDist)
	{
		delete[] weights;
		delete[] onweights;
	}

	if( coarNum>1 )
	{
		delete coarse;

		for( int k=0; k<coarNum; k++ )
			delete coarSet[k];
		delete coarSet;
		delete coarSize;
		delete coarLabel;
	}

	if (pronumforclasses)
		delete[] pronumforclasses;
	if (procenters)
		delete[] procenters;
	if (prodthresh)
		delete[] prodthresh;
	if (frequencies)
	{
		delete[] frequencies;
		frequencies = NULL;
	}
}

// Nearest distance within a class
REAL CLVQ::clsDist( trType* ftr, int dim, int cls, int& index )
{
	REAL dist;
	if( WDist )
		index = nearest( ftr, dim, centers+cls*proNum*dim, proNum, weights+cls*proNum*dim, 
		dthresh+cls*proNum, dist );
	else
		index = nearest( ftr, dim, centers+cls*proNum*dim, proNum, NULL, dthresh+cls*proNum, dist );
	index += cls*proNum;

	return dist;
}

// Return class nearest distances and assigned class label
int CLVQ::classify( trType* input, int dim, REAL* output , int LL)
{
	// distance of nearest prototype in each class
	int ci;
	for (ci = 0; ci < classNum; ci++)
	{
		if (LL == 0)
		{
			if (WDist)
				nearest(input, dim, centers + ci*proNum*dim, proNum, weights + ci*proNum*dim,
				dthresh + ci*proNum, output[ci]);
			else
				nearest(input, dim, centers + ci*proNum*dim, proNum, NULL, dthresh + ci*proNum, output[ci]);
		}
		else
		{
			if (WDist)
				nearest(input, dim, procenters + ci*maxpronum*dim, pronumforclasses[ci], onweights + ci*maxpronum*dim,
				prodthresh + ci*maxpronum, output[ci]);
			else
				nearest(input, dim, procenters + ci*maxpronum*dim, pronumforclasses[ci], NULL, prodthresh + ci*maxpronum, output[ci]);
		}
	}

	REAL mini = (REAL)1E12;
	int index;
	for( ci=0; ci<classNum; ci++ )
	{
		if( output[ci]<mini )
		{
			mini = output[ci];
			index = ci;
		}
	}

	return index;
}

// Search the nearest prototype from a local codebook, for REAL vector
int CLVQ::REnearest( REAL* input, int dim, REAL* vect, int vNum, REAL* thresh, REAL& dmin )
{
	REAL dist, diff;
	int i;

	if( vNum==1 )
	{
		dmin = 0;
		for( i=0; i<dim; i++ )
		{
			diff = (REAL)input[i]-vect[i];
			dmin += diff*diff;
		}
		if( thresh )
			dmin -= thresh[0];

		return 0;
	}

	dmin = (REAL)1E12;
	int m, index;
	for( m=0; m<vNum; m++ )
	{
		if( thresh )
			dist = -thresh[m];
		else
			dist = 0;
		for( i=0; i<dim; i++ )
		{
			diff = (REAL)input[i]-vect[m*dim+i];
			dist += diff*diff;		// square Euclidean distance
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

// Return classes of minimum distances
void CLVQ::nearClassify( trType* input, int dim, int* index, REAL* dmin, int rankNum )
{
	int nearNum;	// number of nearest prototypes to search
	nearNum = proNum*rankNum;

	int* nearIdx;
	REAL* neard;
	if( proNum>1 )
	{
		nearIdx = new int [nearNum];
		neard = new REAL [nearNum];
	}
	else
	{
		nearIdx = index;
		neard = dmin;
	}

	int actRank;
	if( coarNum>1 )		// search with coarse clusters
		actRank = coarseSearch( input, dim, nearIdx, neard, nearNum );
	else
		actRank = nearSearch( input, dim, centers, cenNum, weights, dthresh, nearIdx, neard, nearNum );

	if( proNum>1 )
	{
		// Identify rankNum classes from nearNum nearest prototypes
		int rn = 0;
		for( int k=0; k<actRank; k++ )
		{
			nearIdx[k] /= proNum;		// convert prototype index to class index
			if( !numInList( nearIdx[k], index, rn ) )
			{
				index[rn] = nearIdx[k];
				dmin[rn] = neard[k];
				rn ++;

				if( rn==rankNum )
					break;
			}
		}

		delete nearIdx;
		delete neard;
	}
}

int CLVQ::numInList( int num, int* list, int len )
{
	for( int k=0; k<len; k++ )
	{
		if( list[k]==num )
			return 1;
	}
	return 0;
}

void CLVQ::writeParam( FILE* fp )
{
	fwrite( &classNum, sizeof(int), 1, fp );
	fwrite( &proNum, sizeof(int), 1, fp );
	fwrite( &dim, sizeof(int), 1, fp );
	fwrite( &WDist, sizeof(int), 1, fp );

	fwrite( centers, sizeof(REAL), cenNum*dim, fp );
	if( WDist )
		fwrite( weights, sizeof(REAL), cenNum*dim, fp );

	fwrite( dthresh, sizeof(REAL), cenNum, fp );
	fwrite( &variance, sizeof(REAL), 1, fp );

	fwrite( &coarNum, sizeof(int), 1, fp );
	if( coarNum )
	{
		fwrite( coarse, sizeof(REAL), coarNum*dim, fp );
		fwrite( coarSize, sizeof(int), coarNum, fp );
		for( int k=0; k<coarNum; k++ )
			fwrite( coarSet[k], sizeof(int), coarSize[k], fp );
	}
}

void CLVQ::LVQtrain( trType* data, int sampNum, int dim, int* truth,
			char* configr, REAL regu0, int iteration, REAL relRate, int seed )
{
	// Initialization by clustering
	if( initial==0 )
	{
		srand( seed );
		variance = initProto( data, sampNum, dim, truth );
	}
	if( configr[0]=='M' || configr[0]=='m' )	// no LVQ
		return;

	if( configr[0]=='E' || configr[0]=='e' )
	{
		MCE_LOGM(data, sampNum, dim, truth, regu0, iteration, relRate, 0);
	}
	else if( configr[0]=='L' || configr[0]=='l' )
	{
		printf( "Maximum log-likelihood of margin (LOGM)\n" );
		MCE_LOGM(data, sampNum, dim, truth, regu0, iteration, relRate, 1);
	}
	else if( configr[0]=='W' || configr[0]=='w' )
	{
		printf( "LOGM with weighted distance metric (WLOGM)\n" );
		WLOGM( data, sampNum, dim, truth, regu0, iteration, relRate );
	}

	for (int ci = 0; ci < classNum; ci++)
	{
		for (int i = 0; i < proNum; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				procenters[(ci*maxpronum + i)*dim + j] = centers[(ci*proNum + i)*dim + j];
			}
		}
	}

}


REAL CLVQ::LVQtest(trType* data, int sampNum, int dim, int cnum, int* truth, int LL)
{
	int i, j, k;
	REAL rnum, res;
	trType *input = new trType[dim];
	memset(input, 0, sizeof(trType)*dim);
	REAL *output = new REAL[cnum];
	memset(output, 0, sizeof(REAL)*dim);
	rnum = 0;

	for (i = 0; i < sampNum; i++)
	{
		for (j = 0; j < dim; j++)
		{
			input[j] = data[i*dim + j];
		}
		
		k = classify(input, dim, output, LL);
		if (k != truth[i])
		{
			rnum = rnum + 1;
		}
	}

	if (input != NULL)
	{
		delete[] input;
	}
	if (output != NULL)
	{
		delete[] output;
	}

	res = rnum / sampNum;
	return res;
}


// LOG margin with weighted distance metric
void CLVQ::WLOGM( trType* data, int sampNum, int dim, int* truth, REAL regu0,
			   int iteration, REAL relRate )
{
	REAL regu;		// normalized regularization coefficient
	regu = regu0/variance;

	REAL rate0, rate;
	REAL V, xi;

	V = variance/2;
	xi = 1/V;
	rate0 = (REAL)0.1*relRate/xi;		// initial learning rate

	int n, cls;
	int errNum, last;
	int index, rival;	// closest prototype from genuine and rival class
	REAL dmin1, dmin2;
	REAL margin, lkh;
	REAL coef;
	REAL rate2;		// for updating weights

	int coarIdx, coarRvl;	// index of genuine and rival coarse center
	REAL coarD1, coarD2;

	for( int cycle=0; cycle<=iteration; cycle++ )
	{
		errNum = 0;
		for( n=0; n<sampNum; n++ )
		{
			rate = rate0-rate0*(cycle*sampNum+n)/(iteration*sampNum);

			cls = truth[n];
			if( cls<0 )		// outlier samples excluded
				continue;

			index = nearest( data+(_int64)n*dim, dim, centers+cls*proNum*dim, proNum,
				weights+cls*proNum*dim, dthresh+cls*proNum, dmin1 );
			index += cls*proNum;		// nearest prototype from genuine class
			
			rival = nearRival( data+(_int64)n*dim, dim, cls, dmin2);

			if( dmin1>dmin2 )		// misclassification
				errNum ++;

			if( cycle<iteration)	// last cycle only for retrieval
			{
				margin = dmin2-dmin1;
				if( margin>2/xi )		// no updating
					continue;

				lkh = sigmoid( margin*xi );
				coef = (1-lkh)*xi;
				rate2 = (REAL)0.1*rate*dim/variance;
				modify( data+(_int64)n*dim, dim, centers+index*dim, weights+index*dim, (coef+regu)*rate, (coef+regu)*rate2 );
				modify( data+(_int64)n*dim, dim, centers+rival*dim, weights+rival*dim, -coef*rate, -coef*rate2 );
			}
		}
		/**
		if( (cycle+1)%(iteration/10)==0 || iteration-cycle<=10 )
			printf( "cycle %2d: %5d\n", cycle, errNum );
		**/

		last = errNum;
	}
}

// MCE: Juang & Katagiri, IEEE TSP, 1992
// LOGM: log-likelihood loss of margin
void CLVQ::MCE_LOGM(trType* data, int sampNum, int dim, int* truth, REAL regu0,
	int iteration, REAL relRate, int LL)
{
	REAL regu;		// normalized regularization coefficient
	regu = regu0/variance;

	REAL rate0, rate;
	REAL V, xi;

	V = variance/2;
	xi = 1/V;
	if( LL )
		rate0 = (REAL)0.1*relRate/xi;		// initial learning rate for LOGM
	else
		rate0 = (REAL)0.2*relRate/xi;		// initial learning rate for MCE

	int n, cls;
	int errNum, last;
	int index, rival;	// closest prototype from genuine and rival class
	REAL dmin1, dmin2;
	REAL measure, loss;
	REAL lkh, coef;

	int coarIdx, coarRvl;
	REAL coarD1, coarD2;

	for( int cycle=0; cycle<=iteration; cycle++ )
	{
		errNum = 0;
		for( n=0; n<sampNum; n++ )
		{
			rate = rate0-rate0*(cycle*sampNum+n)/(iteration*sampNum);

			cls = truth[n];
			if( cls<0 )		// outlier samples excluded
				continue;

			index = nearest( data+(_int64)n*dim, dim, centers+cls*proNum*dim, proNum, 
				NULL, dthresh+cls*proNum, dmin1 );
			index += cls*proNum;		// nearest prototype from genuine class
			
			/**
			if( coarNum>1 )
			{
				coarIdx = coarLabel[index];		// label of coarse cluster
				rival = coarseRival( data+(_int64)n*dim, dim, cls, dmin2, 
					coarIdx, coarD1, coarRvl, coarD2 );
			}
			else**/
				rival = nearRival( data+(_int64)n*dim, dim, cls, dmin2 );

			if( dmin1>dmin2 )		// misclassification
				errNum ++;

			if( cycle<iteration)	// last cycle only for retrieval
			{
				measure = dmin1-dmin2;
				if( measure<-2/xi )		// no updating
					continue;

				if( LL )
				{
					lkh = sigmoid( -measure*xi );	// margin=-measure
					coef = (1-lkh)*xi;
				}
				else
				{
					loss = sigmoid( measure*xi );
					coef = loss*(1-loss)*xi;
				}
				modify( data+(_int64)n*dim, dim, centers+index*dim, (coef+regu)*rate );
				modify( data+(_int64)n*dim, dim, centers+rival*dim, -coef*rate );
			}
		}
		if( (cycle+1)%(iteration/10)==0 || iteration-cycle<=10 )
			printf( "cycle %2d: %5d\n", cycle, errNum );

		last = errNum;
	}
}

////////////////////////////////////////////////////////////////////////////////////

// Draw a prototype vector toward a input vector (rate>0), or draw away (rate<0)
void CLVQ::modify( trType* input, int dim, REAL* ref, REAL rate )
{
	for( int i=0; i<dim; i++ )
		ref[i] += rate*( input[i]-ref[i] );
}

// Update prototype with weights
void CLVQ::modify( trType* input, int dim, REAL* ref, REAL* wgt, REAL rate, REAL r2 )
{
	REAL diff;
	REAL wsum = 0;
	for( int i=0; i<dim; i++ )
	{
		diff = input[i]-ref[i];
		ref[i] += rate*wgt[i]*wgt[i]*( diff );
		wgt[i] -= r2*wgt[i]*diff*diff;
		wsum += wgt[i]*wgt[i];
	}
	for( int i=0; i<dim; i++ )
		wgt[i] = sqrt( wgt[i]*wgt[i]*dim/wsum );	// weight normalization
}

void CLVQ::modify( trType* input, int dim, int proIdx, REAL rate )
{
	REAL* ref;
	ref = centers+proIdx*dim;
	for( int i=0; i<dim; i++ )
		ref[i] += rate*( input[i]-ref[i] );
}

// Compute the distances to all prototypes
void CLVQ::distance( trType* input, int dim, REAL* dists )
{
	REAL diff;
	int m, i;

	for( m=0; m<cenNum; m++ )
	{
		dists[m] = -dthresh[m];
		for( i=0; i<dim; i++ )
		{
			diff = centers[m*dim+i]-input[i];
			if( WDist )
				dists[m] += weights[m*dim+i]*weights[m*dim+i]*diff*diff;
			else
				dists[m] += diff*diff;		// square Euclidean distance
		}
	}
}

REAL CLVQ::sigmoid(REAL input)
{
	REAL output = 0.0;

	output = 1 / (1+ exp(-1*input)); 
	return output;
}

// Search multiple nearest prototypes from a given codebook
// return the actual number of ranks
int CLVQ::nearSearch( trType* input, int dim, REAL* centers, int cenNum, REAL* wgt,
					 REAL* thresh, int* index, REAL* dmin, int rankNum )
{
	int ri;
	for( ri=0; ri<rankNum; ri++ )
	{
		index[ri] = -1;
		dmin[ri] = (REAL)1E12+ri;
	}

	REAL dist, diff;
	int m, i;
	int pos;

	for( m=0; m<cenNum; m++ )
	{
		if( thresh )
			dist = -thresh[m];
		else
			dist = 0;
		for( i=0; i<dim; i++ )
		{
			diff = (REAL)input[i]-centers[m*dim+i];
			if( wgt )
				dist += wgt[m*dim+i]*wgt[m*dim+i]*diff*diff;
			else
				dist += diff*diff;		// square Euclidean distance
			if( dist>=dmin[rankNum-1] )
				break;
		}

		if( dist<dmin[rankNum-1] )
		{
			pos = posAscd( dist, dmin, rankNum );
			for( ri=rankNum-1; ri>pos; ri-- )
			{
				dmin[ri] = dmin[ri-1];
				index[ri] = index[ri-1];
			}
			dmin[pos] = dist;
			index[pos] = m;
		}
	}

	int rn = rankNum;
	while( index[rn-1]<0 )
		rn --;

	return rn;
}

// Hieraichical search, return the actual number of ranks
int CLVQ::coarseSearch( trType* input, int dim, int* index, REAL* dmin, int rankNum )
{
	int* coarIdx;
	REAL* coarDmin;
	coarIdx = new int [coarRank];
	coarDmin = new REAL [coarRank];

	// coarse centers have no weights and thresholds
	nearSearch( input, dim, coarse, coarNum, NULL, NULL, coarIdx, coarDmin, coarRank );

	int ri;
	for( ri=0; ri<rankNum; ri++ )
	{
		index[ri] = -1;
		dmin[ri] = (REAL)1E12+ri;
	}

	REAL dist, diff;
	int m, i;
	int k, n;
	int pos;

	for( int c=0; c<coarRank; c++ )
	{
		if( coarDmin[c]-coarDmin[0]>2*variance )		// pruning
			break;

		k = coarIdx[c];
		for( n=0; n<coarSize[k]; n++ )
		{
			m = coarSet[k][n];
			dist = -dthresh[m];		// prototype-specific threshold
			for( i=0; i<dim; i++ )
			{
				diff = (REAL)input[i]-centers[m*dim+i];
				if( WDist )
					dist += weights[m*dim+i]*weights[m*dim+i]*diff*diff;
				else
					dist += diff*diff;		// square Euclidean
				if( dist>=dmin[rankNum-1] )
					break;		// NOT much effective
			}

			if( dist<dmin[rankNum-1] )
			{
				pos = posAscd( dist, dmin, rankNum );
				for( ri=rankNum-1; ri>pos; ri-- )
				{
					dmin[ri] = dmin[ri-1];
					index[ri] = index[ri-1];
				}
				dmin[pos] = dist;
				index[pos] = m;
			}
		}
	}

	if (coarIdx != NULL)
	{
		delete[] coarIdx;
	}
	if (coarDmin != NULL)
	{
		delete[] coarDmin;
	}

	int rn = rankNum;
	while( index[rn-1]<0 )
		rn --;

	return rn;
}

// Rank position in an ordered array, by bisection search
int CLVQ::posAscd( REAL dist, REAL* dmin, int candiNum )
{
	if( dist<dmin[0] || candiNum<=1 )
		return 0;

	int b1, b2, pos;

	b1 = 0;
	b2 = candiNum-1;
	while( b2-b1>1 )	// bi-section search
	{
		pos = (b1+b2)/2;
		if( dist<dmin[pos] )
			b2 = pos;
		else
			b1 = pos;
	}
	return b2;
}

// Search the nearest prototype from a local codebook
int CLVQ::nearest( trType* input, int dim, REAL* vect, int vNum, REAL *wgt, REAL* thresh, REAL& dmin )
{
	REAL dist, diff;
	int i;

	if( vNum==1 )
	{
		dmin = 0;
		for( i=0; i<dim; i++ )
		{
			diff = (REAL)input[i]-vect[i];
			if( wgt )
				dmin += wgt[i]*wgt[i]*diff*diff;
			else
				dmin += diff*diff;
		}
		if( thresh )
			dmin -= thresh[0];

		return 0;
	}

	dmin = (REAL)1E12;
	int m, index;
	for( m=0; m<vNum; m++ )
	{
		if( thresh )
			dist = -thresh[m];
		else
			dist = 0;
		for( i=0; i<dim; i++ )
		{
			diff = (REAL)input[i]-vect[m*dim+i];
			if( wgt )
				dist += wgt[m*dim+i]*wgt[m*dim+i]*diff*diff;
			else
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

// Search the nearest rival prototype, excluding the genuine class
int CLVQ::nearRival( trType* input, int dim, int cls, REAL& dmin )
{
	REAL dist, diff;
	int m, i, ci;
	int index;

	dmin = (REAL)1E12;
	for( m=0; m<cenNum; m++ )
	{
		ci = m/proNum;
		if( ci==cls )		// of genuine class
			continue;

		dist = -dthresh[m];
		for( i=0; i<dim; i++ )
		{
			diff = (REAL)input[i]-centers[m*dim+i];
			if( WDist )
				dist += weights[m*dim+i]*weights[m*dim+i]*diff*diff;
			else
				dist += diff*diff;		// Euclidean distance
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

// Search for the nearest rival prototype from coarse clusters
int CLVQ::coarseRival( trType* input, int dim, int cls, REAL& dmin,
				int coarCls, REAL& coarD1, int& coarRvl, REAL& coarD2 )
{
	int* coarIdx;
	REAL* coarDmin;
	int k, i;
	REAL diff;

	coarIdx = new int [coarRank];		// multiple cluster candidates
	coarDmin = new REAL [coarRank];

	// number of coarse cluster candidates: coarRank
	// coarse centers have no weights and threshold
	nearSearch( input, dim, coarse, coarNum, NULL, NULL, coarIdx, coarDmin, coarRank );
	if( coarIdx[0]==coarCls )
	{
		coarRvl = coarIdx[1];	// closest rival
		coarD2 = coarDmin[1];
		coarD1 = coarDmin[0];	// distance of coarCls
	}
	else
	{
		coarRvl = coarIdx[0];	// closest rival
		coarD2 = coarDmin[0];

		for( k=1; k<coarRank; k++ )
		{
			if( coarIdx[k]==coarCls )
				break;
		}
		if( k<coarRank )	// coarCls in cluster candidates
			coarD1 = coarDmin[k];
		else
		{
			coarD1 = 0;
			for( i=0; i<dim; i++ )
			{
				diff = coarse[coarCls*dim+i]-input[i];
				coarD1 += diff*diff;
			}
		}
	}

	REAL dist;
	int m, ci;
	int index;
	int ri, n;

	// search for rival prototype from the coarse clusters
	dmin = (REAL)1E12;
	for( ri=0; ri<coarRank; ri++ )
	{
	/**	
	for test
	if (coarDmin[ri] - coarDmin[0]>2 * variance)		// pruning threshold: 2*var
			break;**/

		k = coarIdx[ri];
		for( n=0; n<coarSize[k]; n++ )		// prototypes in a coarse cluster
		{
			m = coarSet[k][n];
			ci = m/proNum;
			if( ci==cls )		// of genuine class
				continue;

			dist = -dthresh[m];		// prototypes have thresholds
			for( i=0; i<dim; i++ )
			{
				diff = (REAL)input[i]-centers[m*dim+i];
				if( WDist )
					dist += weights[m*dim+i]*weights[m*dim+i]*diff*diff;
				else
					dist += diff*diff;		// Euclidean distance
				if( dist>=dmin )
					break;
			}

			if( dist<dmin )
			{
				dmin = dist;
				index = m;
			}
		}
	}

	if (coarIdx != NULL)
	{
		delete[] coarIdx;
	}
	if (coarDmin != NULL)
	{
		delete[] coarDmin;
	}

	return index;
}

////////////////////////////////////////////////////////////////////////////////////

// Clustering training data for each class to initialize the prototypes
REAL CLVQ::initProto( trType* data, int sampNum, int dim, int* truth )
{
	CCluster* pClust;
	pClust = new CCluster;

	int* csnum;		// number of samples of each class
	csnum = new int [classNum];
	memset( csnum, 0, classNum*sizeof(int) );

	int n;
	for( n=0; n<sampNum; n++ )
	{
		if( truth[n]>=0 )
			csnum[ truth[n] ] ++;
	}

	int* index;
	trType* tdata;		// class data
	int tnum;

	REAL vari = 0;
	for( int ci=0; ci<classNum; ci++ )	// for each class
	{
		index = new int [ csnum[ci] ];
		tdata = new trType [ csnum[ci]*dim ];

		tnum = 0;
		for( n=0; n<sampNum; n++ )
		{
			if( truth[n]==ci )
			{
				memcpy( tdata+(_int64)tnum*dim, data+(_int64)n*dim, dim*sizeof(trType) );
				tnum ++;
			}
		}
		vari += pClust->cluster( tdata, tnum, dim, centers+ci*proNum*dim, proNum, index );

		if (tdata!= NULL)
		{
			delete[] tdata;
		}
		if (index != NULL)
		{
			delete[] index;
		}
	}
	vari /= classNum;		// average of all classes
	printf( "variance: %6.1f\n", vari );

	for( int k=0; k<cenNum; k++ )
		dthresh[k] = 0;		// thresholds initially as 0

	if( WDist )
	{
		for( int i=0; i<cenNum*dim; i++ )
			weights[i] = 1;

		for (int i = 0; i < oncenNum*dim; i++)
			onweights[i] = 1;
	}

	delete[] csnum;
	delete pClust;

	for (int ci = 0; ci < classNum; ci++)
	{
		for (int i = 0; i < proNum; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				procenters[(ci*maxpronum + i)*dim + j] = centers[(ci*proNum + i)*dim + j];
			}
		}
	}

	for (int i = 0; i < classNum; i++)
	{
		pronumforclasses[i] = proNum;
	}

	for (int ci = 0; ci < classNum; ci++)
	{
		for (int i = 0; i < pronumforclasses[ci]; i++)
		{
			frequencies[(ci*maxpronum) + i] = -1;
		}
	}

	for (int k = 0; k<classNum*maxpronum; k++)
		prodthresh[k] = 0;

	return vari;
}

// Update prototype vector only
void CLVQ::REmodify( REAL* redftr, int redDim, REAL* ref, REAL rate )
{
	for( int j=0; j<redDim; j++ )
		ref[j] += rate*( redftr[j]-ref[j] );
}

void CLVQ::OnlineTrain(trType* singledata, int label, char* configr, REAL thresh, int counts, REAL relRate, REAL regu){

	if (configr[0] == 'M' || configr[0] == 'm')	// no LVQ
		return;

	if (configr[0] == 'E' || configr[0] == 'e')
	{
		Online_MCE_LOGM(singledata, label, thresh, relRate, regu, 0);
	}
	else if (configr[0] == 'L' || configr[0] == 'l')
	{
		printf("Maximum log-likelihood of margin (LOGM)\n");
		Online_MCE_LOGM(singledata, label, thresh, relRate, regu, 1);
	}
	else if (configr[0] == 'W' || configr[0] == 'w')
	{
		printf("LOGM with weighted distance metric (WLOGM)\n");
		Online_WLOGM(singledata, label, thresh, counts, relRate, regu);
	}
}

// Search the nearest rival prototype, excluding the genuine class
int CLVQ::nearRivalPart(trType* input, int dim, int cls, REAL& dmin)
{
	REAL dist, diff;
	int m, i, k,ci;
	int index;

	dmin = (REAL)1E12;
	for (m = 0; m < classNum; m++)
	{
		if (m == cls)
			continue;

		for (i = 0; i < pronumforclasses[m]; i++)
		{
			ci = m*maxpronum + i;
			dist = -prodthresh[ci];
			for (k = 0; k < dim; k++)
			{
				diff = (REAL)input[k] - procenters[ci*dim + k];
				if (WDist)
				{
					dist += onweights[ci*dim + k] * onweights[ci*dim + k] * diff*diff;
				}
				else
				{
					dist += diff*diff;
				}

				if (dist >= dmin)
					break;
			}

			if (dist < dmin)
			{
				dmin = dist;
				index = ci;
			}
		}

	}

	return index;
}

void CLVQ::Online_MCE_LOGM(trType* singledata, int label, REAL thresh, REAL rate, REAL regu, int LL)
{
	int i, j, k, num = 0;
	int index, rival;
	REAL dmin1, dmin2;
	///REAL regu;		// normalized regularization coefficient
	///regu = 0.05 / variance;

	///REAL rate0, rate;
	REAL V, xi;

	V = variance / 2;
	xi = 1 / V;
	//rate = (REAL)0.1 / xi;
	REAL lkh, coef;
	REAL measure, loss;
	//rate0 = (REAL)0.1*relRate / xi;

	if (label < 0)		// outlier samples excluded
		return;

	num = pronumforclasses[label];

	index = nearest(singledata, dim, procenters + label*maxpronum*dim, num,
		NULL, prodthresh + label*maxpronum, dmin1);

	index += label*maxpronum;		// nearest prototype from genuine class

	rival = nearRivalPart(singledata, dim, label, dmin2);
	
	///Add prototype for online by yyshen
	if ((dmin1 > thresh) && num < maxpronum)
	{
		for (i = 0; i < dim; i++)
		{
			procenters[(label*maxpronum + num)*dim + i] = singledata[i];
		}

		pronumforclasses[label] += 1;
	}
	else
	{
		measure = dmin1 - dmin2;
		if (measure < -2 / xi)		// no updating
			return;

		if (LL)
		{
			lkh = sigmoid(-measure*xi);	// margin=-measure
			coef = (1 - lkh)*xi;
		}
		else
		{
			loss = sigmoid(measure*xi);
			coef = loss*(1 - loss)*xi;
		}

		modify(singledata, dim, procenters + index*dim, (coef + regu)*rate);
		modify(singledata, dim, procenters + rival*dim, -coef*rate);
	}
}

// LOG margin with weighted distance metric
void CLVQ::Online_WLOGM(trType* singledata, int label, REAL thresh1, int counts, REAL rate, REAL regu)
{
	int i, j, k, num = 0;
	///REAL regu, regu0 = 0.05; // normalized regularization coefficient
	///regu = regu0 / variance;

	///REAL rate0, rate;
	REAL V, xi;

	V = variance / 2;
	xi = 1 / V;
	///rate0 = (REAL)0.1*relRate / xi;		// initial learning rate

	int index, rival;	// closest prototype from genuine and rival class
	REAL dmin1, dmin2;
	REAL margin, lkh;
	REAL coef;
	REAL rate2;		// for updating weights

	int coarIdx, coarRvl;	// index of genuine and rival coarse center
	REAL coarD1, coarD2;

	if (label < 0)		// outlier samples excluded
		return;

	num = pronumforclasses[label];

	index = nearest(singledata, dim, procenters + label*maxpronum*dim, num, onweights + label*maxpronum*dim, prodthresh + label*maxpronum, dmin1);
	index += label*maxpronum;		// nearest prototype from genuine class
	if ((frequencies[index] > 0) && (frequencies[index] < counts))
	{
		frequencies[index]++;
		if (frequencies[index] == counts)
			frequencies[index] = -1;

		return;
	}

	rival = nearRivalPart(singledata, dim, label, dmin2);
	margin = dmin2 - dmin1;

	///Add prototype for online by yyshen
	///if ((dmin1 > thresh1 || abs(margin) < thresh2) && num < maxpronum)
	if ((dmin1 > thresh1) && num < maxpronum)
	{
		for (i = 0; i < dim; i++)
		{
			procenters[(label*maxpronum + num)*dim + i] = singledata[i];
		}

		frequencies[label*maxpronum + num] = 1;
		if (frequencies[index] == counts)
			frequencies[index] = -1;
		pronumforclasses[label] += 1;
	}
	else
	{
		margin = dmin2 - dmin1;
		if (margin > 2)		// no updating
			return;

		lkh = sigmoid(margin*xi);
		coef = (1 - lkh)*xi;
		rate2 = (REAL)0.1*rate*dim / variance;
		modify(singledata, dim, procenters + index*dim, onweights + index*dim, (coef + regu)*rate, (coef + regu)*rate2);
		modify(singledata, dim, procenters + rival*dim, onweights + rival*dim, -coef*rate, -coef*rate2);
	}
}

