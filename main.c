#include <stdio.h>
#include "matrix.h"

int main(int argc, char** argv)
{
	matrix *m = newMat(2,3);

	printf("%ld, %ld\n", m->height, m->width);

	initMat(m, 1);
	printMat(m);
	printf("\n\n");

	for(int i = 0; i < 2; i++)
	{
		for(int j = 0; j < 3; j++)
			setMat(m, i, j, 10*i+j);
	}
	printMat(m);

	freeMat(m);

	return 0;
}
