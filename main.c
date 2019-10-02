#include <stdio.h>
#include "matrix.h"

int main(int argc, char** argv)
{
	matrix *m1 = newMat(2,1), *m2 = newMat(1,2), *m3 = NULL, *m4 = NULL;
	
	initMat(m1, 1);
	initMat(m2,2);
	m3 = dotMat(m1, m2);
	m4 = dotMat(m2, m1);

	printMat(m1);
	printf("\n\n");
	printMat(m2);
	printf("\n\n");
	printMat(m3);
	printf("\n\n");
	printMat(m4);
	printf("\n\n");


	freeMat(m1);
	freeMat(m2);
	freeMat(m3);
	freeMat(m4);

	return 0;
}
