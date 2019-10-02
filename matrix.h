#ifndef MATRIX_H
#define MATRIX_H

typedef struct
{
	size_t height, width;
	double *mat;
}matrix;

matrix *newMat(int h, int w);
void freeMat(matrix *m);
double getMat(matrix *m, size_t line, size_t col);
void setMat(matrix *m, size_t line, size_t col, double val);
void initMat(matrix *m, double val);
void printMat(matrix *m);

#endif //MATRIX_H
