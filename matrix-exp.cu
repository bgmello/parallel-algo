/*
Fast matrix exponentiation using CUDA.
The exectuable takes two arguments: the matrix in a file and the power.
The file should have a integer N in the first line that is the size of the matrix and
N lines with N integers each representing the matrix.
Example of file:
3
1 2 3
4 5 6
7 8 9

Example of usage:
./mExp input.txt 4
*/
#include<fstream>

typedef long long ll;

using namespace std;

void read_matrix(ifstream& file, ll* matrix, int n){
  int i=0;
  while(file >> matrix[i++]){}
}

int get_size_warp(){
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  return props.multiProcessorCount;
}

__global__ void multi_matrix(ll *a, ll *b, ll *c, int n){

  int tidx = threadIdx.x+blockDim.x*blockIdx.x, slidex = blockDim.x*gridDim.x;
  int tidy = threadIdx.y+blockDim.y*blockIdx.y, slidey = blockDim.y*gridDim.y;
  for(int i=tidx;i<n;i+=slidex){
    for(int j=tidy;j<n;j+=slidey){
      ll tmp=0;
      for(int k=0;k<n;k++){
        tmp+=a[i*n+k]*b[k*n+j];
      }
      c[i*n+j]=tmp;
    }
  }
}


void exp_matrix(ll *a, int n, int exp){

  ll *idt;
  ll *m1_device, *m2_device, *m3_device, *a_device;
  ll *tmp;
  int size = n*n*sizeof(ll);
  int size_warp = get_size_warp();
  dim3 grid(16, 16), block(size_warp, size_warp);

  idt = (ll*)malloc(size);

  cudaMalloc(&m1_device, size);
  cudaMalloc(&m2_device, size);
  cudaMalloc(&m3_device, size);
  cudaMalloc(&a_device, size);

  for(int i=0;i<n*n;i++)idt[i]=0;
  for(int i=0;i*(n+1)<n*n;i++)idt[i*(n+1)]=1;

  cudaMemcpy(a_device, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(m1_device, idt, size, cudaMemcpyHostToDevice);
  cudaMemcpy(m3_device, idt, size, cudaMemcpyHostToDevice);

  while(exp!=0){
    int b = exp%2;
    multi_matrix<<<grid, block>>>(a_device, m1_device, m2_device, n);
    cudaDeviceSynchronize();
    if(b){
      multi_matrix<<<grid, block>>>(m3_device, m2_device, m1_device, n); 
      cudaDeviceSynchronize();
      tmp = m1_device;
      m1_device = m2_device;
      m3_device = tmp;
    }
    else m1_device = m2_device;

    exp>>=1;
  }

  cudaMemcpy(a, m3_device, size, cudaMemcpyDeviceToHost);
}

int main(int argc, char **argv){

  if(argc<3)return 0;

  ifstream inFile(argv[1]);
  ofstream outFile(string(argv[1])+".output");

  int exp = stoi(argv[2]);
  int n;
  ll *matrix;

  inFile >> n;

  matrix = (ll*)malloc((n*n)*sizeof(ll));

  read_matrix(inFile, matrix, n);

  exp_matrix(matrix, n, exp);

  cudaDeviceSynchronize();

  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      outFile << matrix[i*n+j] << " ";
    }
    outFile << endl;
  }
  return 0;
}
