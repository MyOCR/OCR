#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

matrix *newmat(int w, int h)
{
	matrix m = malloc (sizeof(matrix));
	m.mat = malloc (sizeof(float)*w*h);
	m.width = w;
	m.height = h;
	return m;
}
