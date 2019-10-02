#ifndef MATRIX_H
#define MATRIX_H

typedef struct
{
	size_t height, width;
	double *mat;
}matrix;

matrix *newMat(size_t h, size_t w);
void freeMat(matrix *m);
double getMat(matrix *m, size_t line, size_t col);
void setMat(matrix *m, size_t line, size_t col, double val);
void initMat(matrix *m, double val);
matrix *addMat(matrix *m1, matrix *m2);
matrix *dotMat(matrix *m1, matrix *m2);
/*TODO
 * negMat
 * subMat
 * copyMat
 */


void printMat(matrix *m);

#endif //MATRIX_H
