#include <stdlib.h>
#include <stdio.h>
#include "function.h"

int main(int argc, char** argv)
{
	int n = 5;

	printf("Hello World\n");

	for(int i = 0; i < argc; i++)
	{
		printf("arg num %d : %s\n", i, argv[i]);
	}

	printf("fact(%d) = %d\n", n, factRec(n));

	return 0;
}
