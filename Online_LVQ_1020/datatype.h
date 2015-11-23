//#define BYTEVER
#define FLOATVER

#ifdef FLOATVER
typedef float dType;
typedef float trType;	// transformed feature data type
#else
#ifdef SHORTVER
typedef unsigned char dType;
typedef short trType;	// transformed feature data type
#else
typedef unsigned char dType;
typedef char trType;	// BYTEVER
#endif
#endif

typedef float REAL;
