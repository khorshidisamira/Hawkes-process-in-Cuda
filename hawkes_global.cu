#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <math.h>
#include <cassert>
// includes, project
using namespace std;

bool run(float *host_samples);
static void CheckCudaError(const char *, unsigned, const char *, cudaError_t);

/* prototype for ERROR handling */
#define CUDA_CHECK_RETURN(value) CheckCudaError (__FILE__,__LINE__, #value, value)

const int DATA_SIZE = 1;
const int DB_SIZE = 25;

const char* OUTPUT = "outputTimes.txt";
const char* FILENAME = "LockeLowell.csv";

/*
* Check the return value of the CUDA runtime API call and exit the application if the call has failed
*/
static void CheckCudaError(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr
		<< statement
		<< " returned "
		<< cudaGetErrorString(err)
		<< "(" << err << ") at "
		<< file << ":"
		<< line
		<< std::endl;

	exit(EXIT_FAILURE);
}

// the result should be lower triangular shaped!!
__global__ void pjiCalculator(float *data, unsigned int dbsize) {
	int row = blockDim.x*blockIdx.x + threadIdx.x;
	int col = blockDim.y*blockIdx.y + threadIdx.y;
	int dimx = blockDim.x*gridDim.x;
	int gtid = col*dimx + row;
	 
	int ii = 0;
	for (int k = 0; k <= gtid; k++) {
		ii = k*dbsize + gtid;
		data[ii] = 0.0f;
		data[ii] = 0.0f;
	}

	int id = 0; 
	for (int j = gtid + 1; j < dbsize; j++) {
		id = j*dbsize + gtid; 
		data[id] = 2.0f; // for test!
	} 
}



__device__ void col_sum(float * arr, float * res, unsigned int gindex, unsigned int db_size){
	int id = 0;
	//int size = db_size - gindex - 1;
	float *column_elements;
	for (int j = gindex + 1; j < db_size; j++) {
		id = j*db_size + gindex;
		column_elements[id] = arr[id];
	}
	atomicAdd(res, *column_elements);

}

__global__ void k0Kernel(float *data, float *result, unsigned int dbsize) {
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int dimx = blockDim.x*gridDim.x;
	int gtid = iy*dimx + ix;

	printf("GTID: %d,\n", gtid);

	float * partial_result;
	col_sum(data, &(partial_result[gtid]), gtid, dbsize);
	__syncthreads();
	if(gtid == 0) {
		atomicAdd(result, *partial_result);//sum(pji)
		float s = *result; 
		printf("TOTAL SUM iS:%f", s); 
	} 
}

float summation(float *arr, int size) {
	float sum_val = 0.0f;
	for (int h = 0; h < size; h++) {
		for (int w = 0; w < size; w++){
			sum_val += arr[DB_SIZE * h + w] ;
		}
	}
	return sum_val;
}

void test_k0Kernel(float *h_data, int DB_SIZE, float k0_h_result) {
	float * ret_val = new float;
	float direct_return = summation(h_data, DB_SIZE);

	assert(direct_return == k0_h_result);
	assert(ret_val != nullptr);
}

bool run(float *host_samples) {
	float *h_data = new float[DB_SIZE*DB_SIZE];
	float *d_data = NULL;

	for (int h = 0; h < DB_SIZE; h++) {
		for (int w = 0; w < DB_SIZE; w++)
			if (h > w) {
				h_data[DB_SIZE * h + w] = 1.0f;
			}
			else {
				h_data[DB_SIZE * h + w] = 0.0f;
			}
	}


	//THESE ARE NOT OPTIMIZED(lower triangular is used only)
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_data, DB_SIZE * DB_SIZE * sizeof(float)));

	// copy data directly to device memory
	CUDA_CHECK_RETURN(cudaMemcpy(d_data, h_data, DB_SIZE * DB_SIZE * sizeof(float), cudaMemcpyHostToDevice));

	/* define kernel block info */
	dim3 gridDim(1, 1, 1);
	dim3 blkDim(5, 5, 1);

	//while ((mu_diff > threshold) || (k0_diff > threshold) || (w_diff > threshold)) {
	// Copying back the results from GPU
	
	float *k0_d_result = NULL;
	float k0_h_result = 0.0f;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&k0_d_result, sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(k0_d_result, &k0_h_result, sizeof(float), cudaMemcpyHostToDevice));

	/////////////////////////////E step///////////////////////////////// 
	pjiCalculator << <gridDim, blkDim >> > (d_data, DB_SIZE);
	CUDA_CHECK_RETURN(cudaMemcpy(&h_data, d_data, DB_SIZE * DB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

	k0Kernel << <gridDim, blkDim >> > (d_data, k0_d_result, DB_SIZE);
	CUDA_CHECK_RETURN(cudaMemcpy(&k0_h_result, k0_d_result, sizeof(float), cudaMemcpyDeviceToHost));
	
	
	test_k0Kernel(h_data, DB_SIZE, k0_h_result);

	cudaThreadExit();
	//pjjCalculator << <blockCounts, threadCounts >> > (d_data, DB_SIZE, mu_v, k0_v, w_v);

	/////////////////////////////M step/////////////////////////////////
 
	//} 
 
	 

	// cleanup

	//CPU MEM

	//GPU MEM
	CUDA_CHECK_RETURN(cudaFree(d_data));
	CUDA_CHECK_RETURN(cudaFree(k0_d_result));

	return true;
}

int main(void) {

	printf("Estimating Hawkes process using EM\n");

	//bool bResult = false;
	float *input_samples = new float[DB_SIZE];

	cudaDeviceReset();

	//readDatabase(input_samples);
	run(input_samples);

	//test_func_a();
	//test_lambda_func(input_samples);
	//test_initializeData(input_samples);
	return 0;
}
