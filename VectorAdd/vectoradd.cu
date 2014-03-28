#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <sys/time.h>

const int inputLength = 4096;
const int BLOCK_SIZE = 512;

__global__ void vecAdd(float * in1, float * in2, float * out, int len) 
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if(idx < len)
		out[idx] = in1[idx] + in2[idx];
}

int main(int argc, char ** argv) 
{
	int inputLength, size;
    	float * hostInput1;
    	float * hostInput2;
    	float * hostOutput;
    	float * deviceInput1;
    	float * deviceInput2;
    	float * deviceOutput;

	struct timeval t1, t2;
	size = inputLength * sizeof(float);

        hostInput1 = (float *) malloc(size);
        hostInput2 = (float *) malloc(size);
        hostOutput = (float *) malloc(size);

	cudaMalloc((void **) &deviceInput1, size);
  	cudaMalloc((void **) &deviceInput2, size);
  	cudaMalloc((void **) &deviceOutput, size);


    	cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  	cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

	dim3 DimGrid((inputLength-1)/BLOCK_SIZE + 1, 1, 1);
  	dim3 DimBlock(BLOCK_SIZE, 1, 1);
    
    	gettimeofday(&t1, NULL);
  	vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    	cudaThreadSynchronize();
    	gettimeofday(&t2, NULL);
	printf("Time taken in computing vector sum is %d usec\n", (t2.sec - t1.sec)*1000000 + (t2.usec - t1.usec));	
    	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

    	cudaFree(deviceInput1);
  	cudaFree(deviceInput2);
  	cudaFree(deviceOutput);

    	free(hostInput1);
    	free(hostInput2);
    	free(hostOutput);

    	return 0;
}

