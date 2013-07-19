// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
// Due Tuesday, January 22, 2013 at 11:59 p.m. PST


#define BLOCK_SIZE 1024 //@@ You can change this


__global__ void scan(float * input, float * output, int len) {

  	__shared__ float sharedMem[BLOCK_SIZE];
	
	unsigned int t = threadIdx.x + blockDim.x * blockIdx.x;
//	unsigned int start = 2 * blockIdx.x * blockDim.x;

  	sharedMem[threadIdx.x] = (t < len) ? input[t] : 0;
//	sharedMem[t] = (t < len) ? input[start + t] : 0;
//	sharedMem[blockDim.x + t] = ((blockDim.x + t) < len) ? input[start + blockDim.x + t] : 0;

  	__syncthreads();
  
  	//@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
  int stride = 1;
  while (stride < BLOCK_SIZE) {
   	int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < BLOCK_SIZE) {
     	sharedMem[index] += sharedMem[index - stride];
    }
    
    stride = stride * 2;
    
    __syncthreads();  
  }
  
  for (stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
   	__syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index + stride < BLOCK_SIZE) {
     	sharedMem[index + stride] += sharedMem[index]; 
    }
    
  }
  __syncthreads();
  output[threadIdx.x + blockDim.x * blockIdx.x] = sharedMem[threadIdx.x];
  __syncthreads();
  
/*  if(blockIdx.x > 0) {
	sharedMem[blockIdx.x] = sharedMem[]
    output[threadIdx.x + blockDim.x * blockIdx.x] += output[BLOCK_SIZE - 1]; 
  }
*/
  
  if (threadIdx.x < blockIdx.x) {
//  for (stride = 0; stride < blockIdx.x; stride++) {
//   	sharedMem[stride] = output[stride * blockDim.x + BLOCK_SIZE - 1];
	sharedMem[threadIdx.x] = output[threadIdx.x * blockDim.x + BLOCK_SIZE - 1];
//    __syncthreads();
    
  }
  __syncthreads();
  for (stride = 0; stride < blockIdx.x; stride++) {
   	output[threadIdx.x + blockDim.x * blockIdx.x] += sharedMem[stride];
    __syncthreads();
    
  }
 
  
      
}


int main(int argc, char ** argv) {
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    hostOutput = (float*) malloc(numElements * sizeof(float));


    cudaMalloc((void**)&deviceInput, numElements*sizeof(float));
    cudaMalloc((void**)&deviceOutput, numElements*sizeof(float));

    cudaMemset(deviceOutput, 0, numElements*sizeof(float));

    cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here

  	dim3 DimGrid((numElements - 1)/BLOCK_SIZE + 1, 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);

    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce

    scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);

  cudaDeviceSynchronize();

    cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);


    free(hostInput);
    free(hostOutput);

    return 0;
}

