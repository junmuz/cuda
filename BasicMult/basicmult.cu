// MP 2: Due Sunday, Dec 16, 2012 at 11:59 p.m. PST

// Compute C = A * B
      
__global__ void matrixMultiply(float * A, float * B, float * C,
			       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    // Insert code to implement matrix multiplication here
	
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
  	float temp = 0;

  if((Row < numCRows) && (Col < numCColumns)) { 

	for(int m = 0; m < numAColumns; ++m) {
		temp += A[Row * numAColumns + m] * B[m * numBColumns + Col];
      
    }
    
    C[Row * numCColumns + Col] = temp;
  }
}

int main(int argc, char ** argv) {
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)


    // Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    // Allocate the hostC matrix
    int sizeA = numARows * numAColumns * sizeof(float);
  	int sizeB = numBRows * numBColumns * sizeof(float);
    int sizeC = numCRows * numCColumns * sizeof(float);
    hostC = (float *) malloc(sizeC);  
    // Allocate GPU memory here
	cudaMalloc((void **) &deviceA, sizeA);
  	cudaMalloc((void **) &deviceB, sizeB);
  	cudaMalloc((void **) &deviceC, sizeC);

    // Copy memory to the GPU here

    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  	cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
    
    // Initialize the grid and block dimensions here
    
    dim3 dimGrid(ceil(((float)numCColumns)/16.0), ceil(((float) numCRows)/16.0), 1);
  	dim3 dimBlock(16, 16, 1);
    matrixMultiply<<<dimGrid,dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaThreadSynchronize();
    
    // Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);


    // Free the GPU memory here
    cudaFree(deviceA);
  	cudaFree(deviceB);
  	cudaFree(deviceC);



    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

