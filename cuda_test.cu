#include "common.h"
#include "cuda_common.h"

inline void my_mmset_rand(void *p,int size)
{
        float * tmp=(float *)malloc(size);
        //assert(size%4==0);
        for(int i=0;i<size/4;i++)
                tmp[i]= (rand()%1000)/ 1000.0;
        cudaMemcpy(p,tmp,size,cudaMemcpyHostToDevice);
        free(tmp);
}
inline void my_mmset(void *p,int size,float value)
{
        float * tmp=(float *)malloc(size);
        //assert(size%4==0);
        for(int i=0;i<size/4;i++)
                tmp[i]=value;
        cudaMemcpy(p,tmp,size,cudaMemcpyHostToDevice);
        free(tmp);
}

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

void my_test2()
{
	/*
printf("111\n");
cuda_hello<<<1,1>>>(); 
cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
cudnnFilterDescriptor_t filterDesc;
cudnnActivationDescriptor_t actiDesc;
cudnnConvolutionDescriptor_t convDesc;
cudnnConvolutionFwdAlgo_t fwdAlgo;

cudnnCreateTensorDescriptor(&inputTensor);
cudnnCreateTensorDescriptor(&biasTensor);
cudnnCreateTensorDescriptor(&outputTensor);
*/
}

const int N = 16; 
const int blocksize = 16; 
 
__global__ 
void hello(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}
 
int my_test()
{
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);
 
	printf("%s", a);
 
	cudaMalloc( (void**)&ad, csize ); 
	cudaMalloc( (void**)&bd, isize ); 
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 
	
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
	cudaFree( ad );
	cudaFree( bd );
	
	printf("%s\n", a);
	return EXIT_SUCCESS;
}
