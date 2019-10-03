#ifndef NET_H
#define NET_H

#include "matrix.h"

typedef struct
{
	matrix layer[];
	matrix biases[];
}network;

#endif //NET_H
