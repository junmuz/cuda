// MP 3: Due Sunday, Dec 30, 2012 at 11:59 p.m. PST
// Compute C = A * B
#define TILE_WIDTH 2
#define TILES 2.0
      
__global__ void matrixMultiplyShared(__shared__ float * A, __shared__ float * B, __shared__ float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {

  
	__shared__ float As[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	for(int m = 0; m < numAColumns/TILE_WIDTH; ++m) {
		As[ty][tx] = A[(Row * numAColumns) + (m * TILE_WIDTH + tx)];
		Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
	
		__syncthreads();
		for(int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue += As[ty][k] * Bs[k][tx]; 
		}
		__syncthreads();
	}
	C[Row * numCColumns + Col] = Pvalue;

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


    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    int sizeA = numARows * numAColumns * sizeof(float);
    int sizeB = numBRows * numBColumns * sizeof(float);
    int sizeC = numCRows * numCColumns * sizeof(float);
    hostC = (float *) malloc(sizeC);  
  
    //@@ Allocate GPU memory here

    cudaMalloc((void **) &deviceA, sizeA);
    cudaMalloc((void **) &deviceB, sizeB);
    cudaMalloc((void **) &deviceC, sizeC);
  
    //@@ Copy memory to the GPU here

    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
    
    //@@ Initialize the grid and block dimensions here
  	dim3 dimGrid(ceil(((float)numCColumns)/TILES), ceil(((float) numCRows)/TILES), 1);
  	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  
    //@@ Launch the GPU Kernel here

    matrixMultiplyShared<<<dimGrid,dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaThreadSynchronize();
    //@@ Copy the GPU memory back to the CPU here

    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
  

    //@@ Free the GPU memory here

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);


    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

