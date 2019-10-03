#include <stdio.h>
#include <err.h>
#include "net.h"

network *newNet(size_t numLayer, size_t numNodeForLayer[])
{//create a new network with numLayer layers (input layer include)
	if(numLayer < 2)
		errx(1, "Error in newNet. A network has to have at least two layers.");
	for(int i = 0; i < numLayer; i++)
	{
		if(numNodeForLayer[i] == 0)
			errx(1, "Error in newNet. A layer has to have at least one node");
	}

	network *net = malloc (sizeof(net));

	return network;
}
