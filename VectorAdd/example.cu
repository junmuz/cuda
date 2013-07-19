// MP 1

#define IN_LEN		2048

__global__ void vecAdd(int * out, int len) 
{
    //@@ Insert code to implement vector addition here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if(idx < len)
	out[idx] = idx;
}

int main(int argc, char ** argv) 
{
	int size;
    	int * hostOutput;
    	int * deviceOutput;


    	hostOutput = (int *) malloc(IN_LEN * sizeof(int));

  
	size = IN_LEN * sizeof(int);
  	cudaMalloc((void **) &deviceOutput, size);


  	cudaMemcpy(deviceOutput, hostOutput, size, cudaMemcpyHostToDevice);


	dim3 DimGrid((IN_LEN-1)/1024 + 1, 1, 1);
  	dim3 DimBlock(1024, 1, 1);
    
    
  	vecAdd<<<DimGrid, DimBlock>>>(deviceOutput, IN_LEN);

    	cudaThreadSynchronize();
    
    	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

  	cudaFree(deviceOutput);

    	free(hostOutput);

    	return 0;
}

