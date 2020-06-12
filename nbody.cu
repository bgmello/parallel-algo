/*
Final project of NVIDIA Fundamentals of CUDA in C/C++
Consits of a simulation of the n-body problem.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Do not modify this function. A constraint of this exercise is
 * that it remain a host function.
 */

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */
__global__
void bodyForce(Body *p, float dt, int n) {
  int tidx = threadIdx.x + blockDim.x*blockIdx.x;
  int slidex = blockDim.x*gridDim.x;
  for (int i = tidx; i < n; i+=slidex) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

__global__
void integratePosition(Body *p, float dt, int n){

    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int slide = blockDim.x*gridDim.x;
    for(int i=tid;i<n;i+=slide){
        p[i].x += p[i].vx*dt;
        p[i].y += p[i].vy*dt;
        p[i].z += p[i].vz*dt;
    }

}

int main(const int argc, const char** argv) {

  /*
   * Do not change the value for `nBodies` here. If you would like to modify it,
   * pass values into the command line.
   */

  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;

  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);

  int size_warps = props.multiProcessorCount;

  cudaMallocHost(&buf, bytes);

  Body *p;
  
  cudaMalloc(&p, bytes);

  /*
   * As a constraint of this exercise, `randomizeBodies` must remain a host function.
   */

  randomizeBodies(buf, 6 * nBodies); // Init pos / vel data
  
  cudaMemcpy(p, buf, bytes, cudaMemcpyHostToDevice);

  clock_t start = clock();

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  /*******************************************************************/
  // Do not modify these 2 lines of code.

  for (int iter = 0; iter < nIters; iter++) {
    /*******************************************************************/

    /*
     * You will likely wish to refactor the work being done in `bodyForce`,
     * as well as the work to integrate the positions.
     */


    dim3 num_threads(128);
    dim3 num_blocks(size_warps*8);

    bodyForce<<<num_blocks, num_threads>>>(p, dt, nBodies); // compute interbody forces
    cudaDeviceSynchronize();

    /*
     * This position integration cannot occur until this round of `bodyForce` has completed.
     * Also, the next round of `bodyForce` cannot begin until the integration is complete.
     */
    integratePosition<<<num_blocks, num_threads>>>(p, dt, nBodies); // integrate position
    
    cudaDeviceSynchronize();

  }
  
  double totalTime = (double)(clock()-start)/CLOCKS_PER_SEC;

  cudaMemcpy(buf, p, bytes, cudaMemcpyDeviceToHost);

  double avgTime = totalTime / (double)(nIters);
  double billionsOfOpsPerSecond = (1e-9 * nBodies * nBodies) / (avgTime);

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
  /*******************************************************************/

  /*
   * Feel free to modify code below.
   */

  cudaFree(p);
  cudaFreeHost(buf);
}

