#include <stdlib.h>
#include <stdio.h>
//#include "function.h"

int fact(int n)//return n!
{
	for(int i = n-1; i>0; i--)
	{
		n *= i;
	}

	return n;
}

int factRec(int n)//return n!
{
	return n<=0? 1: n*fact(n-1);
}
