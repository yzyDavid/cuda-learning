#include <iostream>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <ctime>

#include "cuda_runtime.h"

const unsigned int MAX_THREADS_PER_BLOCK = 1024;
const size_t DATA_COUNTS = 50 * 1024 * 1024;

int generated_random_numbers(int *arr, size_t counts)
{
	srand(static_cast<unsigned int>(time(0)));
	for (size_t i = 0; i < counts; i++)
		arr[i] = rand() % 10;
	return 0;
}

void __global__ worker_function(int *data, size_t count, int64_t *result)
{
	int64_t sum = 0;
	for(size_t i = 0; i < count; i++)
		sum += data[i] * data[i];
	*result = sum;
}

void start_work()
{
	auto pData = new int[DATA_COUNTS];
	int *gpuData = nullptr;

	generated_random_numbers(pData, DATA_COUNTS);
	cudaMalloc((void **)(&gpuData), sizeof(int) * DATA_COUNTS);
	cudaMemcpy(gpuData, pData, sizeof(int) * DATA_COUNTS, cudaMemcpyHostToDevice);

	int64_t result;
	int64_t *pgpuResult = nullptr;
	cudaMalloc((void **)&pgpuResult, sizeof(int64_t));
	worker_function<<<1, MAX_THREADS_PER_BLOCK, 10240>>>(gpuData, DATA_COUNTS, pgpuResult);
	cudaMemcpy(&result, pgpuResult, sizeof(int64_t), cudaMemcpyDeviceToHost);
	cudaFree(gpuData);
	cudaFree(pgpuResult);
	std::cout << "RESULT: " << result << std::endl;

	delete pData;
}

int queryCudaDevicesCount()
{
	int count;

	cudaGetDeviceCount(&count);
	if (count == 0)
		return 0;

	return count;
}

int getFirstDeviceProperties()
{
	using namespace std;
	cudaDeviceProp prop;

	if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -1;

	printf("\nMajor version of first CUDA device: %d\n", prop.major);
	printf("Minor version of first CUDA device: %d\n", prop.minor);
	printf("Is ECC Enabled: %d\n", prop.ECCEnabled);
	printf("Async Engine Count: %d\n", prop.asyncEngineCount);
	printf("Can map host memory: %d\n", prop.canMapHostMemory);
	printf("clock rate: %d\n", prop.clockRate);
	cout << "Concurrent Kernels: " << prop.concurrentKernels << endl;
	cout << "Global L1 Cache supported: " << prop.globalL1CacheSupported << endl;
	cout << "Max Threads Dimension: " << prop.maxThreadsDim << endl;
	cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
	cout << "Max Threads per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << endl;
	cout << "MultiProcessorCount: " << prop.multiProcessorCount << endl;
	cout << "Device Name: " << prop.name << endl;
	cout << "Total Global Memory: " << prop.totalGlobalMem << endl;
	cout << endl;
	return 0;
}

int setFirstDevice()
{
	cudaSetDevice(0);
	return 0;
}

int main()
{
	std::printf("Numbers of devices support CUDA on this computer: %d\n", queryCudaDevicesCount());
	assert(getFirstDeviceProperties() == 0);
	setFirstDevice();

	start_work();
	return 0;
}

