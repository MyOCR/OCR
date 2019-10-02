#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "matrix.h"

matrix *newMat(int h, int w)
{//create a matrix of width w, and height h
	matrix *m = malloc (sizeof(matrix));
	m->mat = malloc (sizeof(double)*w*h);
	m->width = w;
	m->height = h;
	return m;
}

void freeMat(matrix *m)
{//free the matrix
	free(m->mat);
	free(m);
}

double getMat(matrix *m, size_t line, size_t col)
{//get the element at line,col
	if(line >= m->height || col >= m->width)
		errx(1, "Error in getMat. There is not any element at this coordinate: line = %ld, col = %ld", line, col);

	return m->mat[line*m->width + col];
}

void setMat(matrix *m, size_t line, size_t col, double val)
{//set val at line,col
	if(line >= m->height || col >= m->width)
		errx(1, "Error in putMat. There is not any space at this coordinate: line = %ld, col = %ld", line, col);

	m->mat[line*m->width+col] = val;
}

void initMat(matrix *m, double val)
{
	for(size_t i = 0; i < m->width*m->height; i++)
		m->mat[i] = val;
}

void printMat(matrix *m)
{
	for(size_t i = 0; i < m->height; i++)
	{
		for(size_t j = 0; j < m->width; j++)
		{
			printf("%08.3lf,\t", getMat(m, i, j));
		}
		printf("\n");
	}
}
