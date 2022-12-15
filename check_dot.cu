#include <iostream>
#include <vector>

using namespace std;
__global__ void Dot2(int* a, int* b, int* out)
{
    int result = a[blockIdx.x] * b[blockDim.x * blockIdx.x + threadIdx.x];
    out[threadIdx.x*gridDim.x+blockIdx.x] = result;

    printf("\n threadID: %d, out: %d, a: %d, b: %d ",threadIdx.x, out[threadIdx.x], a[blockIdx.x], b[blockDim.x * blockIdx.x + threadIdx.x]);
}


void result_dot(int b_size, int t_size, vector<vector<int>> &v, int* out){
    int result = 0;
    for(int i=0; i<t_size; i++){
            for(int j=0; j<b_size; j++){
                result += out[i*3 +j];
            }
            v[0].push_back(result);
            result=0;
    }
}

__global__ void Dot3(int* a, int* b, int* out, int height)
{
    for(int i=0; i<height; i++)
    {
        int result = a[i] * b[i * blockIdx.x + threadIdx.x];
        out[i * blockDim.x + threadIdx.x] = result;
        printf("\n threadID: %d, out: %d, a: %d, b: %d ",threadIdx.x, out[i * blockIdx.x + i], a[i], b[i * blockIdx.x + threadIdx.x]);
    }
}

int a[3] = {1,2,3};
int b[12] = {4,5,6,7,8,9,10,11,12,13,14,15};

int c[10] = {1, 2, 3,4,5,6,7,8,9,10};
int d[10] = {4,5,6,7,8,9,10,11,12,13};


int main()
{
    int* a_dev, *b_dev, *out_dev;
    vector<vector<int>> final;
    int* out = (int*)malloc(sizeof(int)*100);

    cudaMalloc((void**)&a_dev, sizeof(int)*10); // 1* 784
    cudaMalloc((void**)&b_dev, sizeof(int)*10); //784*10
    cudaMalloc((void**)&out_dev, sizeof(int)*100);// 1* 10

    cudaMemcpy(a_dev, c, sizeof(int)*10, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, d, sizeof(int)*10, cudaMemcpyHostToDevice);
    cudaMemcpy(out_dev, out, sizeof(int)*100, cudaMemcpyHostToDevice);

    Dot3 <<<1, 10>>> (a_dev, b_dev, out_dev, 10);
    cudaDeviceSynchronize();
    cudaMemcpy(out, out_dev, sizeof(int)*100, cudaMemcpyDeviceToHost);

    // final.resize(1);
    // result_dot(1, 10, final, out);

    // for(int i=0; i<final[0].size(); i++)
    //     cout << final[0][i] << " ";
    // cout << endl;
    cout << endl;

    for(int i=0; i<10; i++)
    {
        for(int j=0; j<10; j++)
        {
            cout << out[i*10 + j] << " ";
        }
        cout << endl;
    }
                
    return 0;
}
