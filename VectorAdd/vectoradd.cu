// MP 1

__global__ void vecAdd(float * in1, float * in2, float * out, int len) 
{
    //@@ Insert code to implement vector addition here
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


    	hostOutput = (float *) malloc(inputLength * sizeof(float));

  
	size = inputLength * sizeof(float);
	cudaMalloc((void **) &deviceInput1, size);
  	cudaMalloc((void **) &deviceInput2, size);
  	cudaMalloc((void **) &deviceOutput, size);


    	cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  	cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
  	cudaMemcpy(deviceOutput, hostOutput, size, cudaMemcpyHostToDevice);


	dim3 DimGrid((inputLength-1)/1024 + 1, 1, 1);
  	dim3 DimBlock(1024, 1, 1);
    
    
  	vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    	cudaThreadSynchronize();
    
    	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

    	cudaFree(deviceInput1);
  	cudaFree(deviceInput2);
  	cudaFree(deviceOutput);

    	free(hostInput1);
    	free(hostInput2);
    	free(hostOutput);

    	return 0;
}

