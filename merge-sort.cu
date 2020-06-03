#include<stdio.h>
#include<stdlib.h>

__device__
int lower_bound(int *a, int size_a, int x){
  if(x<=a[0])return 0;
  if(a[size_a-1]<x)return size_a;
  int ini=1, end=size_a;
  int mid;
  while(ini<=end){
    mid = (ini+end)/2;
    if(a[mid-1]<x && x<=a[mid])return mid;
    if(a[mid]<=x)ini=mid;
    else end=mid-1;
  }
  return 0;
}

__device__
int upper_bound(int *a, int size_a, int x){
  if(a[size_a-1]<=x)return size_a;
  if(x<a[0])return 0;
  int ini=0, end=size_a-1;
  int mid;
  while(ini<end){
    mid = (ini+end)/2;
    if(a[mid]<=x && x<a[mid+1])return mid+1;
    if(x>=a[mid])ini=mid+1;
    else end=mid;
  }
 return size_a; 
}

__global__
void parallel_merge(int *a, int *c,  int l, int m , int r){
  int pos;
  int tid = threadIdx.x+blockDim.x*blockIdx.x;
  int slide = blockDim.x*gridDim.x;
  for(int i=l+tid;i<m;i+=slide){
    pos = i + upper_bound(a+m, r-m, a[i])-l;
    c[pos+l] = a[i];
  }
  for(int i=m+tid;i<r;i+=slide){
    pos = i + lower_bound(a+l, m-l, a[i])-m;
    c[pos+l] = a[i];
  }
}

void merge_sort(int* arr, int n){
  int *arr_device, *c_device;

  cudaMalloc(&c_device, n*sizeof(int));
  cudaMalloc(&arr_device, n*sizeof(int));
  cudaMemcpy(arr_device, arr, n*sizeof(int), cudaMemcpyHostToDevice);

  int deviceId, warg_size;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  
  warg_size = props.multiProcessorCount;

  int num_threads = 128, num_blocks = 2*warg_size;

  for(int size=1;size<=n-1;size = 2*size){
    for(int l=0;l<n-1;l+=2*size){
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      int m = min(l+size-1, n-1);
      int r = min(l+2*size-1, n-1);
      parallel_merge<<<num_blocks,num_threads, 0, stream>>>(arr_device, c_device, l, m+1, r+1);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(arr_device, c_device, n*sizeof(int), cudaMemcpyDeviceToDevice);
  }
  cudaDeviceSynchronize();
  cudaMemcpy(arr, arr_device, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(arr_device);
  cudaFree(c_device);
}

int main(int argc,char** argv){
  int *v;
  int n;

  if(argc>1)n = 1<<atoi(argv[1]);
  else return 0;

  v = (int*)malloc(n*sizeof(int));
  for(int i=0;i<n;i++)v[i]=rand()%50;
  
  merge_sort(v, n);

  for(int i=0;i<n;i++){
    printf("%d ", v[i]);
  }
  printf("\n");

  free(v);
  return 0;
}
