#include <stdio.h>
#include "datatype.h"
#include "Cluster.h"

////////////////////////////////////////////////////////////////////////////////
// Prototype classifier, learning vector quantization (LVQ)

class CLVQ
{
	int classNum, dim;
	REAL* centers;		// prototype vectors
	int cenNum;		// total number of prototypes
	int proNum;		// prototype number of each class, equal

	REAL* dthresh;	// prototype-specific thresholds

	REAL* weights;	// for weighted distance metric
	int WDist;

	REAL* coarse;		// cluster centers for coarse classification
	int coarNum, coarRank;
	int** coarSet;	// prototypes in each coarse cluster
	int* coarSize;		// number of prototypes in each cluster
	int* coarLabel;	// cluster label of prototypes

	int initial;

	int oncenNum;  //total number of prototypes for online
	int* pronumforclasses;
	int maxpronum;
	trType* procenters;
	REAL* prodthresh;
	REAL* onweights;
	int* frequencies;

	// Compute the distance to all prototypes
	void distance( trType*, int, REAL* );

	REAL sigmoid(REAL);

	// Search multiple nearest prototypes from the global codebook
	int nearSearch( trType*, int, REAL*, int, REAL*, REAL*, int*, REAL*, int );
	int coarseSearch( trType*, int, int*, REAL*, int );
	int posAscd( REAL, REAL*, int );		// position in ascending array
	int REnearest( REAL*, int, REAL*, int, REAL*, REAL& );

	// Search the nearest prototype from a local codebook
	int nearest( trType*, int, REAL*, int, REAL*, REAL*, REAL& );
	// Search the nearest rival prototype
	int nearRival( trType*, int, int cls, REAL&  );
	int coarseRival( trType*, int, int cls, REAL&, int, REAL&, int&, REAL& );

	void modify( trType*, int, REAL*, REAL );	// update a prototype
	void modify( trType*, int, REAL*, REAL*, REAL, REAL );
	void REmodify( REAL*, int, REAL*, REAL );

	REAL initProto( trType*, int, int, int* );
	
	void WLOGM( trType*, int, int, int*, REAL, int, REAL );

	int nearRivalPart(trType*, int, int cls, REAL&);

public:
	///CLVQ( int, int, char*, int );
	CLVQ(int, int, int, char*);
	CLVQ( FILE* );
	~CLVQ();

	REAL variance;	// average variance of all classes
	void MCE_LOGM(trType*, int, int, int*, REAL, int, REAL, int);
	void writeParam( FILE* );
	int classify( trType*, int, REAL* , int);
	REAL clsDist( trType*, int, int, int& );
	void nearClassify( trType*, int, int*, REAL*, int );
	int numInList( int, int*, int );
	void modify( trType*, int, int, REAL );	// update a prototype

	void LVQtrain( trType*, int, int, int*, char*, REAL, int, REAL, int );
	REAL LVQtest(trType*, int, int, int, int *, int);

	void OnlineTrain(trType*, int, char*, REAL, int, REAL, REAL);
	void Online_MCE_LOGM(trType*, int, REAL, REAL,REAL, int);
	void Online_WLOGM(trType*, int, REAL, int, REAL, REAL);
};
