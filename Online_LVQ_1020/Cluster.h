#include "datatype.h"

class CCluster {
	char equal(trType*, trType*, int);	// two feature vectors equal or not
	int nearest(trType*, int, REAL*, int, REAL&);

	void CL(trType*, int, int, REAL*, int, int);		// competitive learning
	
	void DA(trType*, int, int, REAL*, int, int);		// deterministic annealing

	void distance(trType*, int, REAL*, int, REAL*);
	int sensitive(trType*, int, REAL*, int, int*, REAL&);
	void modify(trType*, int, REAL*, REAL);
	REAL errComp(trType*, int, int, REAL*, int);

public:
	void initial(trType*, int, int, REAL*, int);
	REAL kmeans(trType*, int, int, REAL*, int, int*, int);
	REAL FSCL(trType*, int, int, REAL*, int, int);	// frequency-sensitive
	REAL cluster(trType*, int, int, REAL*, int, int*);
};


class CFCluster {
	char equal(REAL*, REAL*, int);	// two feature vectors equal or not

	void CL(REAL*, int, int, REAL*, int, int);		// competitive learning
	
	void DA(REAL*, int, int, REAL*, int, int);		// deterministic annealing

	void distance(REAL*, int, REAL*, int, REAL*);
	int sensitive(REAL*, int, REAL*, int, int*, REAL&);
	void modify(REAL*, int, REAL*, REAL);
	REAL errComp(REAL*, int, int, REAL*, int);

	REAL E2dist(REAL*, REAL*, int);
	void cenMerge(REAL*, REAL*, int, int, int);
	REAL fproxim(REAL*, REAL*, int, int, int);

public:

	void initial(REAL*, int, int, REAL*, int);
	REAL kmeans(REAL*, int, int, REAL*, int, int*, int);

	int nearest(REAL*, int, REAL*, int, REAL&);
	REAL cluster(REAL*, int, int, REAL*, int, int*);
	REAL FSCL(REAL*, int, int, REAL*, int, int);		// frequency-sensitive
	void agglomerative(REAL*, int, int, REAL*, int, int*);
};
