#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define TILE_WIDTH 2
      
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
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
    int numARows = 4; // number of rows in the matrix A
    int numAColumns = 4; // number of columns in the matrix A
    int numBRows = 4; // number of rows in the matrix B
    int numBColumns =4; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C
    int numCColumns; // number of columns in the matrix C


    numCRows = numARows;
    numCColumns = numBColumns;

    int sizeA = numARows * numAColumns * sizeof(float);
    int sizeB = numBRows * numBColumns * sizeof(float);
    int sizeC = numCRows * numCColumns * sizeof(float);

    hostA = (float *) malloc(sizeA);
    hostB = (float *) malloc(sizeB);

    for(int i = 0; i < (numARows * numAColumns); i++) {
        hostA[i] = i;
    }

    for(int i = 0; i < (numBRows * numBColumns); i++) {
        hostB[i] = i;
    }

    hostC = (float *) malloc(sizeC);  
  
    // Allocate GPU memory here

    cudaMalloc((void **) &deviceA, sizeA);
    cudaMalloc((void **) &deviceB, sizeB);
    cudaMalloc((void **) &deviceC, sizeC);
  
    // Copy memory to the GPU here

    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
    
    // Initialize the grid and block dimensions here
  	dim3 dimGrid(ceil(((float)numCColumns)/TILE_WIDTH), ceil(((float) numCRows)/TILE_WIDTH), 1);
  	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  
    //@@ Launch the GPU Kernel here

    matrixMultiplyShared<<<dimGrid,dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaThreadSynchronize();
    //@@ Copy the GPU memory back to the CPU here

    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
  
    for(int i = 0; i < (numCRows * numCColumns); i++) {
        printf("%f\n", hostC[i]);
    }

    //@@ Free the GPU memory here

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);


    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

