#include <iostream>
#include <vector>

using namespace std;
__global__ void Sumb2(int* a, int* b, int* out)
{
    int tid, tx, ty;
    tx = blockDim.x * blockIdx.x + threadIdx.x;
    ty = blockDim.y * blockIdx.y + threadIdx.y;
    tid = gridDim.x * blockDim.x * ty + tx;

    out[tid] = a[tid] + b[tid];
}

__global__ void Sigmoid(float* a, float* out)
{
    int tid, tx, ty;
    tx = blockDim.x * blockIdx.x + threadIdx.x;
    ty = blockDim.y * blockIdx.y + threadIdx.y;
    tid = gridDim.x * blockDim.x * ty + tx;

    out[tid] = 1.0/(1.0+expf(-(a[tid]/4000)));
}

float a[3] = {1,2,3};
float b[3] = {4,5,6};

int main()
{
    float* a_dev, *b_dev, *out_dev;
    vector<vector<int>> final;
    float* out = (float*)malloc(sizeof(float)*3);

    cudaMalloc((void**)&a_dev, sizeof(float)*3); // 1* 784
    cudaMalloc((void**)&b_dev, sizeof(float)*3); //784*10
    cudaMalloc((void**)&out_dev, sizeof(float)*3);// 1* 10

    cudaMemcpy(a_dev, a, sizeof(float)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, sizeof(float)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(out_dev, out, sizeof(float)*3, cudaMemcpyHostToDevice);

    Sigmoid <<<3, 1>>> (a_dev, out_dev);
    cudaDeviceSynchronize();
    cudaMemcpy(out, out_dev, sizeof(float)*3, cudaMemcpyDeviceToHost);


    for(int i=0; i<3; i++)
        cout << out[i] << " ";
    cout << endl;

                
    return 0;
}
