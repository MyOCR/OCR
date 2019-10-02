#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "matrix.h"

matrix *newMat(size_t h, size_t w)
{//create a matrix of width w, and height h
	if(h == 0 || w == 0)
		errx(1, "Error in newMat. A matrix's width and height can not be null");
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
{//initialize each value of m to val
	for(size_t i = 0; i < m->width*m->height; i++)
		m->mat[i] = val;
}

matrix *addMat(matrix *m1, matrix *m2)
{/*
return m1+m2
/!\ it creates a third matrix 
*/
	if(m1->height != m2->height || m1->width != m2->width)
		errx(1, "Error in addMat. The matrix m1 and m2 have different width or different height");

	matrix *m = newMat(m1->height, m1->width);

	for(size_t i = 0; i < m->height*m->width; i++)
		m->mat[i] = m1->mat[i] + m2->mat[i];

	return m;
}

matrix *dotMat(matrix *m1, matrix *m2)
{/*
return m1.m2
/!\ it creates a third matrix 
*/
	if(m1->width != m2->height)
		errx(1, "Error id dotMat. m1's width and m2's height are diffents");

	matrix *m = newMat(m1->height, m2->width);

	for(size_t i = 0; i < m->height; i++)
	{
		for(size_t j = 0; j < m->width; j++)
		{
			double n = 0;
			for(size_t k = 0; k < m1->width; k++)
				n += getMat(m1, i, k) * getMat(m2, k, j);
			setMat(m, i, j, n);
		}
	}

	return m;
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
