#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan(). This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result

__global__ void block_level_scan(int* output, int N, int powN, int* blockSums) { 
    __shared__ int partial_sum[THREADS_PER_BLOCK]; 
    int local_idx = threadIdx.x; 
    int block_begin = blockIdx.x * blockDim.x; 
    int true_idx = block_begin + local_idx; 

    int validThreads = min(blockDim.x, powN - block_begin); 

    if (true_idx >= powN) return; 

    if (true_idx < N) {
        partial_sum[local_idx] = output[true_idx]; 
    }
    else {
        partial_sum[local_idx] = 0; 
    }
    __syncthreads();

    if (blockSums) {
        blockSums[blockIdx.x] = 0; 
    }
    
    // upsweep phase 
    for (int two_d = 1; two_d <= validThreads/2; two_d*=2) { 
        int two_dplus1 = 2*two_d; 
        int idx = (local_idx + 1) * two_dplus1 - 1; 
        if (idx < validThreads) 
            partial_sum[idx] += partial_sum[idx - two_d]; 
        __syncthreads(); 
    }
    
    if (local_idx == validThreads - 1) {
        if (blockSums) {
            blockSums[blockIdx.x] = partial_sum[local_idx]; 
        }
        partial_sum[local_idx] = 0; 
    }

    __syncthreads(); 
    
    // downsweep phase 
    for (int two_d = validThreads/2; two_d >= 1; two_d /= 2) { 
        int two_dplus1 = 2*two_d; 
        int idx = (local_idx + 1) * two_dplus1 - 1; 
        if (idx < validThreads) { 
            int t = partial_sum[idx-two_d]; 
            partial_sum[idx-two_d] = partial_sum[idx]; 
            partial_sum[idx] += t; 
        } 
        __syncthreads(); 
    } 
    if(true_idx < N) 
        output[true_idx] = partial_sum[local_idx]; 
}

__global__ void sum_scan(int* output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (i >= N) return; 

    // upsweep phase
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        int k = (i + 1) * two_dplus1 - 1; 
        if (k < N)
            output[k] += output[k - two_d];
        __syncthreads(); 
    }
 
    
    if(i == 0) {
        output[N-1] = 0; 
    }
    __syncthreads(); 

    // downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d; 
        int k = (i + 1) * two_dplus1 - 1; 
        if (k < N) {
            int t = output[k-two_d];
            output[k-two_d] = output[k];
            output[k] += t;
        }
         __syncthreads(); 
    }
}

__global__ void coalesce_sums(int * output, int n, int* blockSums) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < n)
        output[idx] += blockSums[blockIdx.x]; 
}

void recursive_block_scan(int* results, int N) {

    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    if (blocks == 1) {
        int powN = nextPow2(N); 
        block_level_scan<<<1, powN>>>(results, N, powN, nullptr);
        cudaDeviceSynchronize();
        return;
    }
    
    int* blockSums;
    int powBlocks = nextPow2(blocks); //pad up to 64 blockSums
    cudaMalloc(&blockSums, powBlocks * sizeof(int));
    if (blocks < powBlocks) {
        cudaMemset(blockSums + blocks, 0, (powBlocks - blocks) * sizeof(int));
    }

    int remainder = N & (THREADS_PER_BLOCK-1); 
    int paddedN = (remainder == 0) ? N : (N - remainder + nextPow2(remainder));

    block_level_scan<<<blocks, THREADS_PER_BLOCK>>>(results, N, paddedN, blockSums); 

    cudaDeviceSynchronize();

    int temp[powBlocks]; 
    
    recursive_block_scan(blockSums, blocks);

    cudaMemcpy(temp, blockSums, powBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    coalesce_sums<<<blocks, THREADS_PER_BLOCK>>>(results, N, blockSums);
    cudaDeviceSynchronize();

    cudaMemcpy(temp, blockSums, powBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(blockSums);
}

void exclusive_scan(int* input, int N, int* result)
{
    recursive_block_scan(result, N); 
}

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.
    //for(int i = 0; i < N; i++){printf("original: %d\n", inarray[i]);}
    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired. If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found

__global__ void repeat_pairs(int* input, int* flag, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i >= N-1) return;
    
    if (input[i] == input[i+1]) {
        flag[i] = 1; 
    }
    else {
        flag[i] = 0;
    }
}
__global__ void get_indices(int* input, int* prefix_sums, int* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i >= N-1) return;

    if (input[i] != input[i+1]) return;

    output[prefix_sums[i]] = i;
}

int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    int* flags;
    int rounded_length = nextPow2(length); 
    int cnt; 

    cudaMalloc((void **)&flags, sizeof(int) * rounded_length);

    int active_threads = length; 
    int threadsPerBlock = min(rounded_length, THREADS_PER_BLOCK);
    int blocks = (active_threads + threadsPerBlock - 1) / threadsPerBlock;
    
    repeat_pairs<<<blocks,threadsPerBlock>>>(device_input, flags, length); 

    exclusive_scan(flags, length, flags);

    get_indices<<<blocks,threadsPerBlock>>>(device_input, flags, device_output, length); 
    
    cudaMemcpy(&cnt, flags + (length - 1), sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(flags);

    return cnt; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf(" SMs: %d\n", deviceProps.multiProcessorCount);
        printf(" Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf(" CUDA Cap: %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}


// -------------------------------------------------------------------------
// | Element Count   | Ref Time        | Student Time    | Score           |
// -------------------------------------------------------------------------
// | 1000000         | 1.247           | 0.764           | 1.25            |
// | 10000000        | 13.114          | 3.367           | 1.25            |
// | 20000000        | 21.327          | 9.239           | 1.25            |
// | 40000000        | 42.051          | 11.447          | 1.25            |
// -------------------------------------------------------------------------
// |                                   | Total score:    | 5.0/5.0         |
// -------------------------------------------------------------------------
