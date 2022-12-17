#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <typeinfo>
#include <stdlib.h> 



using namespace std;

vector<vector<float>> load_data(26880);
vector<vector<float>> X_train(26880);
vector<vector<float>> x_val(8400);
vector<vector<float>> x_test(6720);
vector<float> y_train;
vector<float> y_val;
vector<float> y_test;

vector<float> losses, val_losses;
vector<vector<float>> w1(784), w2(10);
vector<float> b1(10), b2(10);
vector<vector<float>> output(10), a1_train(1, vector<float>(10, 0)), a1_test(1, vector<float>(10, 0)), a1_val(1, vector<float>(10, 0));
vector<float> b1_grad(10);
float b2_grad;
vector<vector<float>> w1_grad(10), w2_grad(10); // w2_grad (10, 10)


int otos = 1*784 * sizeof(float);
int stot = 784*10 * sizeof(float);
int otot = 1*10 * sizeof(float);
int ttot = 10*10 * sizeof(float);
int ttoo = 10*1 * sizeof(float);

float* x_arr, *w1_arr, *out_arr, *b1_arr, *a1_arr, *b2_arr, *w2_arr, *sout_arr, *sout2_arr;
float* x_dev, *w1_dev, *out_dev, *b1_dev, *a1_dev, *b2_dev, *w2_dev, *sout_dev, *sout2_dev;
float l1 = 0, l2 = 0;


// 전치행렬 만들기
vector<vector<float>> transpose(vector<vector<float>>& a)
{
    vector<vector<float>> ret(a[0].size(), vector<float>(a.size()));
    for(int i=0;i<a.size();i++)
    {
        for(int j=0;j<a[i].size();j++)
        {
            ret[j][i] = a[i][j];
        }
    }
    return ret;
}


// 행렬 곱 연산 함수
vector<vector<float>> dot(vector<vector<float>> &a, vector<vector<float>>& b)
{
    vector<vector<float>> out(a.size(), vector<float>(b[0].size(), 0.0f));

    for(int i=0;i<a.size();i++)
    {
        for(int j=0;j<b[0].size();j++)
        {
            for(int k=0;k<b.size();k++)
            {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return out;
}

__global__ void Dot2(float* a, float* b, float* out)
{
    float result = a[blockIdx.x] * b[blockDim.x * blockIdx.x + threadIdx.x];
    out[threadIdx.x*gridDim.x+blockIdx.x] = result;
}

__global__ void Dot3(float* a, float* b, float* out, int height)
{
    for(int i=0; i<height; i++)
    {
        float result = a[i] * b[i * blockIdx.x + threadIdx.x];
        out[i * blockDim.x + threadIdx.x] = result;
    }
}

void result_dot(int b_size, int t_size, vector<vector<float>> &v, float* out){
    float result = 0.0f;
    for(int i=0; i<t_size; i++){
        for(int j=0; j<b_size; j++){
            result += out[i*b_size +j];
        }
        v[0].push_back(result);
        result=0.0f;
    }
}


// 행렬 합 연산 함수
__global__ void Sumb2(float* a, float* b, float* out)
{
    int tid, tx, ty;
    tx = blockDim.x * blockIdx.x + threadIdx.x;
    ty = blockDim.y * blockIdx.y + threadIdx.y;
    tid = gridDim.x * blockDim.x * ty + tx;

    out[tid] = a[tid] + b[tid];
}


// sigmoid 함수
__global__ void Sigmoid(float* a, float* out)
{
    int tid, tx, ty;
    tx = blockDim.x * blockIdx.x + threadIdx.x;
    ty = blockDim.y * blockIdx.y + threadIdx.y;
    tid = gridDim.x * blockDim.x * ty + tx;

    out[tid] = 1.0/(1.0+expf(-(a[tid]/4000)));
}


// vector에서 array로 변환하는 함수
void vec2arr(vector<vector<float>> v, float* d)
{
    for(int i=0; i<v.size(); i++)
    {
        for(int j=0; j<v[i].size(); j++)
        {
            d[i*v[0].size() + j] = v[i][j];
        }
    }
}


// array에서 vector로 변환하는 함수
void arr2vec(float* d, vector<vector<float>> &v)
{
    for(int i=0; i<v.size(); i++)
    {
        for(int j=0; j<10; j++)
        {
            v[i].push_back((float)d[i*10 + j]);    
        }
    }
}


// array를 0으로 초기화하는 함수
void arr_init(float* d, int size1, int size2)
{
    for(int i=0; i<size1; i++)
    {
        for(int j=0; j<size2; j++)
        {
            d[i*size2 + j] = 0.0f;
        }
    }
}


// forword 함수
vector<vector<float>> forward(vector<vector<float>> &x, vector<vector<float>> &w1, vector<vector<float>> &w2, vector<vector<float>> &a1){
    vector<vector<float>> out;

    x_arr = (float*)malloc(otos);
    w1_arr = (float*)malloc(stot);
    out_arr = (float*)malloc(stot);
    a1_arr = (float*)malloc(otot);
    b2_arr = (float*)malloc(otot);
    w2_arr = (float*)malloc(ttot);


    // ############## out = dot(x, w1); ##############
    vec2arr(x, x_arr);
    vec2arr(w1, w1_arr);

    memset(out_arr, 0, sizeof(out_arr));

    cudaMalloc((void**)&x_dev, otos);
    cudaMalloc((void**)&w1_dev, stot);
    cudaMalloc((void**)&out_dev, stot);

    cudaMemset(x_dev, 0, otos);
    cudaMemset(w1_dev, 0, stot);
    cudaMemset(out_dev, 0, stot);

    cudaMemcpy(x_dev, x_arr, otos, cudaMemcpyHostToDevice);
    cudaMemcpy(w1_dev, w1_arr, stot, cudaMemcpyHostToDevice);
    cudaMemcpy(out_dev, out_arr, stot, cudaMemcpyHostToDevice);

    Dot2 <<<784, 10>>> (x_dev, w1_dev, out_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(out_arr, out_dev, stot, cudaMemcpyDeviceToHost);
    out.resize(1);
    result_dot(784, 10, out, out_arr);

    cudaFree(x_dev); cudaFree(w1_dev); cudaFree(out_dev);
    free(x_arr); free(w1_arr); free(out_arr);


    // ############## sumb(s1, b1, sout) ##############
    float *s1_arr, *s1_dev;
    s1_arr = (float*)malloc(otot);
    b1_arr = (float*)malloc(otot);
    sout_arr = (float*)malloc(otot);

    vec2arr(out, s1_arr);
    for(int i=0; i<b1.size(); i++)
    {
        b1_arr[i] = b1[i];
    }

    cudaMalloc((void**)&s1_dev, otot);
    cudaMalloc((void**)&b1_dev, otot);
    cudaMalloc((void**)&sout_dev, otot);

    cudaMemcpy(s1_dev, s1_arr, otot, cudaMemcpyHostToDevice);
    cudaMemcpy(b1_dev, b1_arr, otot, cudaMemcpyHostToDevice);

    Sumb2 <<<10, 1>>>(s1_dev, b1_dev, sout_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(sout_arr, sout_dev, otot, cudaMemcpyDeviceToHost);
    out.clear();
    out.resize(1);
    arr2vec(sout_arr, out);

    cudaFree(b1_dev); cudaFree(sout_dev); cudaFree(s1_dev);
    free(b1_arr); free(sout_arr); free(s1_arr);


    // ############## a1 = sigmoid(out); ##############  
    float *sig_arr, *sig_dev, *sigout_arr, *sigout_dev;
    sig_arr = (float*)malloc(otot);
    sigout_arr = (float*)malloc(otot);

    vec2arr(out, sig_arr);

    cudaMalloc((void**)&sig_dev, otot);
    cudaMalloc((void**)&sigout_dev, otot);

    cudaMemcpy(sig_dev, sig_arr, otot, cudaMemcpyHostToDevice);
    cudaMemcpy(sigout_dev, sigout_arr, otot, cudaMemcpyHostToDevice);

    Sigmoid <<<10, 1>>>(sig_dev, sigout_dev);
    cudaDeviceSynchronize();
    cudaMemcpy(sigout_arr, sigout_dev, otot, cudaMemcpyDeviceToHost);
    a1.clear();
    a1.resize(1);
    arr2vec(sigout_arr, a1);
    cudaFree(sig_dev); cudaFree(sigout_dev);
    free(sig_arr); free(sigout_arr);


    // ############## out2 = dot(a1, w2); ##############  
    vec2arr(a1, a1_arr);
    vec2arr(w2, w2_arr);
    
    float* out2_arr = (float*)malloc(ttot);
    memset(out2_arr, 0, sizeof(out2_arr));
    float* out2_dev;
   

    cudaMalloc((void**)&a1_dev, otot);
    cudaMalloc((void**)&w2_dev, ttot);
    cudaMalloc((void**)&out2_dev, ttot);

    cudaMemcpy(a1_dev, a1_arr, otot, cudaMemcpyHostToDevice);
    cudaMemcpy(w2_dev, w2_arr, ttot, cudaMemcpyHostToDevice);
    cudaMemcpy(out2_dev, out2_arr, ttot, cudaMemcpyHostToDevice);

    Dot2 <<<10, 10>>> (a1_dev, w2_dev, out2_dev);
    cudaDeviceSynchronize();
    cudaMemcpy(out2_arr, out2_dev, ttot, cudaMemcpyDeviceToHost);
    
    out.clear();
    out.resize(1);
    result_dot(10, 10, out, out2_arr);

    cudaFree(a1_dev); cudaFree(w2_dev); cudaFree(out2_dev);
    free(a1_arr); free(w2_arr); free(out2_arr);


    // ############## sumb(out, b2, out); ##############
    s1_arr = (float*)malloc(otot);
    b2_arr = (float*)malloc(otot);
    sout_arr = (float*)malloc(otot);

    vec2arr(out, s1_arr);
    for(int i=0; i<b2.size(); i++)
    {
        b2_arr[i] = b2[i];
    }

    cudaMalloc((void**)&s1_dev, otot);
    cudaMalloc((void**)&b2_dev, otot);
    cudaMalloc((void**)&sout_dev, otot);

    cudaMemcpy(s1_dev, s1_arr, otot, cudaMemcpyHostToDevice);
    cudaMemcpy(b2_dev, b2_arr, otot, cudaMemcpyHostToDevice);

    Sumb2 <<<10, 1>>>(s1_dev, b2_dev, sout_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(sout_arr, sout_dev, otot, cudaMemcpyDeviceToHost);
    out.clear();
    out.resize(1);
    arr2vec(sout_arr, out);

    cudaFree(b2_dev); cudaFree(sout_dev); cudaFree(s1_dev);
    free(b2_arr); free(sout_arr); free(s1_arr);
    
    return out;
}


// softmax 함수
vector<vector<float>> softmax(vector<vector<float>> a){
    float total = 0.0f;
    vector<vector<float>> out;

    for(int i=0; i<a.size(); i++)
    {
        out.push_back(vector<float>());
        for(int j=0;j<a[i].size();j++)
        {
            out[i].push_back(exp(a[i][j]));
            total += out[i].back();
        }
    }

    for(int i=0;i<out.size();i++)
    {
        for(int j=0;j<out[i].size();j++)
        {
            out[i][j] /= total;
        }
    }

    return out;
}


// y label one-hot encoding 전처리 함수
vector<vector<float>> y_train_encoded(vector<float> &y){
    vector<vector<float>> output(y.size(), vector<float>(10, 0));

    for(int i=0;i<output.size();i++)
    {
        output[i][int(y[i])] = 1;
    }
    return output;
}


// backpropagation 함수 -> gradiant 계산하기 
void backprop(vector<vector<float>> x, vector<vector<float>> err, vector<vector<float>> &w2, vector<vector<float>> &w1_grad,\
             vector<vector<float>> &w2_grad, vector<float> &b1_grad, float &b2_grad, vector<vector<float>> &a1){
    for(int i=0; i<err.size(); i++)
    {
        b2_grad = 0.0f;
        for(int j=0; j<err[i].size(); j++)
        {
            b2_grad += err[i][j];
        }
    }

    vector<vector<float>> a1T = transpose(a1);


    // ############## w2_grad = dot(a1T, err); ##############
    float *a1T_arr, *a1T_dev, *err_arr, *err_dev, *w2g_arr, *w2g_dev;
    a1T_arr = (float*)malloc(ttoo);
    err_arr = (float*)malloc(otot);
    w2g_arr = (float*)malloc(ttot);

    vec2arr(a1T, a1T_arr);
    vec2arr(err, err_arr);
    memset(w2g_arr, 0, sizeof(w2g_arr));

    cudaMalloc((void**)&a1T_dev, ttoo);
    cudaMalloc((void**)&err_dev, otot);
    cudaMalloc((void**)&w2g_dev, ttot);

    cudaMemset(a1T_dev, 0, ttoo);
    cudaMemset(err_dev, 0, otot);
    cudaMemset(w2g_dev, 0, ttot);

    cudaMemcpy(a1T_dev, a1T_arr, ttoo, cudaMemcpyHostToDevice);
    cudaMemcpy(err_dev, err_arr, otot, cudaMemcpyHostToDevice);
    cudaMemcpy(w2g_dev, w2g_arr, ttot, cudaMemcpyHostToDevice);

    Dot3 <<<1, 10>>> (a1T_dev, err_dev, w2g_dev, 10);
    cudaDeviceSynchronize();

    cudaMemcpy(w2g_arr, w2g_dev, ttot, cudaMemcpyDeviceToHost);
    w2_grad.clear();
    w2_grad.resize(10);
    arr2vec(w2g_arr, w2_grad);

    cudaFree(a1T_dev); cudaFree(err_dev); cudaFree(w2g_dev);
    free(a1T_arr); free(err_arr); free(w2g_arr); 


    vector<vector<float>> w2T = transpose(w2);
    vector<vector<float>> err_to_hidden = dot(err, w2T);

    for(int i=0; i<a1.size(); i++)
    {
        for(int j=0; j<a1[i].size(); j++)
        {
            err_to_hidden[i][j] *= a1[i][j] * (1-a1[i][j]);
        }
    }

    vector<vector<float>> xT = transpose(x);
    w1_grad = dot(xT, err_to_hidden);
    for(int j=0; j<err_to_hidden[0].size(); j++)
    {
        b1_grad[j] = 0.0f;
        for(int i=0; i<err_to_hidden.size(); i++)
        {
            b1_grad[j] += err_to_hidden[i][j];
        }
    }
}


// training 함수
vector<vector<float>> training(vector<vector<float>> x, vector<vector<float>> y, vector<vector<float>> &w1, \
                    vector<vector<float>> &w2, vector<vector<float>> &w1_grad, vector<vector<float>> &w2_grad, \
                    vector<float> &b1, vector<float> &b2, vector<float> &b1_grad, \
                    float &b2_grad, vector<vector<float>> &a1)
{   
    vector<vector<float>> z = forward(x, w1, w2, a1);                               // forward 진행

    vector<vector<float>> a = softmax(z);                                           // forward 진행한 값을 softmax 수행

    // cross entropy로 오차 구하기
    vector<vector<float>> err(y.size());
    for(int i=0; i<y.size(); i++)
    {
        for(int j=0; j<y[i].size(); j++)
        {
            err[i].push_back(-(y[i][j]-a[i][j]));
        }
    }

    backprop(x, err, w2, w1_grad, w2_grad, b1_grad, b2_grad, a1);                   // backpropagation 진행

    // 계산한 gradiant를 가지고 weight와 bias 업데이트
    vector<vector<int>> sign1(w1.size(), vector<int>(w1[0].size(), 0));
    vector<vector<int>> sign2(w2.size(), vector<int>(w2[0].size(), 0));
    for(int i=0; i<w1.size(); i++)
    {
        for(int j=0; j<w1[i].size(); j++)
        {
            sign1[i][j] = (fabs(w1[i][j]) < 1e-9) ? 0 : round(w1[i][j]/(fabs(w1[i][j])));
        }
    }
    for(int i=0; i<w2.size(); i++)
    {
        for(int j=0; j<w2[i].size(); j++)
        {
            sign2[i][j] = (fabs(w2[i][j]) < 1e-9) ? 0 : round(w2[i][j]/(fabs(w2[i][j])));
        }
    }
    
    float lr = 0.1;

    for(int i=0; i<w1_grad.size(); i++)
    {
        for(int j=0; j<w1_grad[i].size(); j++)
        {
            w1_grad[i][j] += (l1*sign1[i][j] + l2*w1[i][j]);
        }
    }
    for(int i=0; i<w2_grad.size(); i++)
    {
        for(int j=0; j<w2_grad[i].size(); j++)
        {
            w2_grad[i][j] += (l1*sign2[i][j] + l2*w2[i][j]);
        }
    }

    for(int i=0;i<w1.size();i++)
    {
        for(int j=0;j<w1[i].size();j++)
        {
            w1[i][j] -= lr * w1_grad[i][j];
        }
    }
    for(int i=0;i<w2.size();i++)
    {
        for(int j=0;j<w2[i].size();j++)
        {
            w2[i][j] -= lr * w2_grad[i][j];
        }
    }

    for(int i=0;i<b1.size();i++)
    {
        b1[i] -= lr * b1_grad[i];
    }
    for(int i=0;i<b1.size();i++)
    {
        b2[i] -= lr * b2_grad;
    }

    return a;
}


// regulation 함수
float reg_loss(vector<vector<float>> w1, vector<vector<float>> w2)
{
    float tmp1 = 0.0f;

    for(int i=0; i<w1[0].size(); i++)
    {
        for(int j=0; j<w1.size(); j++)
        {
            if(w1[j][i] < 0) w1[j][i] = -w1[j][i];
            tmp1 += l1*(w1[j][i]) + l2/2*pow(w1[j][i],2);
        }

        for(int j=0; j<w2[i].size(); j++)
        {
            if(w2[i][j] < 0) w2[i][j] = -w2[i][j];
            tmp1 += w2[i][j] + pow(w2[i][j],2);
        }
    }

    return tmp1;
}


// validation 진행 함수
void update_val_loss(vector<vector<float>> x_val, vector<vector<float>> y_val, vector<vector<float>> &w1, vector<vector<float>> &w2, \
                    vector<float> &b1, vector<float> &b2, vector<vector<float>> &a1)
{
    vector<vector<float>> z = forward(x_val, w1, w2, a1);           // forward 진행

    vector<vector<float>> a = softmax(z);                           // forward 진행한 값을 softmax 수행
    
    float val_loss = 0.0f;
    for(int ii=0;ii<a.size();ii++)
    {
        for(int jj=0;jj<a[ii].size();jj++)
        {
            val_loss += (-y_val[ii][jj] * log(a[ii][jj]));
        }
    }
    val_losses.push_back((val_loss+reg_loss(w1, w2))/y_val.size());  // loss 저장
}


// fit 함수 -> training + update_val_loss
void fit(vector<vector<float>> &x_val, vector<vector<float>> &y_val, vector<vector<float>> &x, vector<vector<float>> &y, int epochs){
    vector<vector<float>> trained_a(10);
    vector<float> loss(trained_a.size());
    for(int i=0; i<epochs; i++)
    {
        float loss = 0.0f;
        cout << ".";

        vector<vector<float>> smallx(1, vector<float>(784, 0.0)), smally(1, vector<float>(10, 0));  // data 한 개씩 불러오기
        int x_sizee = x.size();
        for(int j=0; j<x_sizee; j++)
        {
            smallx[0].assign(x[j].begin(), x[j].end());
            smally[0].assign(y[j].begin(), y[j].end());

            trained_a = training(smallx, smally, w1, w2, w1_grad, w2_grad, b1, b2, b1_grad, b2_grad, a1_train);  // training 진행 

            float loss = 0.0f;
            for(int ii=0;ii<trained_a.size();ii++)
            {
                for(int jj=0;jj<trained_a[ii].size();jj++)
                {
                    loss += (-y_val[ii][jj] * log(trained_a[ii][jj]));
                }
            }
        }

        losses.push_back((loss+reg_loss(w1, w2))/smally.size());

        // validation 진행
        for(int j=0;j<x_val.size();j++)
        {
            smallx[0].assign(x_val[j].begin(), x_val[j].end());
            smally[0].assign(y_val[j].begin(), y_val[j].end());
            update_val_loss(smallx, smally, w1, w2, b1, b2, a1_train);
        }       
    }
    cout << endl;
}


// weight initailization 함수
void kaiming_init(vector<vector<float>> &w, int n_in){
    float std = sqrt(2/(float) n_in);
    
    random_device rd;
    mt19937 gen(rd()); 
    normal_distribution<float> dist(0.0f, std); 

    for (int i=0; i<n_in; i++){
        for(int j=0; j<10; j++)
        {
            w[i].push_back(dist(gen));
        }
    }
}


// 학습한 모델로 예측값 도출하는 함수
vector<int> predict(vector<vector<float>> x, vector<vector<float>> &w1, vector<vector<float>> &w2){
    vector<int> result;
    vector<vector<float>> smallx(1);

    for(int i=0; i<x.size(); i++)
    {
        smallx[0].assign(x[i].begin(), x[i].end());
        vector<vector<float>> z = forward(smallx, w1, w2, a1_train);

        float tmp = -1e9;
        int idx = 0;
        for(int j=0; j<z[0].size(); j++)
        {
            if(tmp < z[0][j])
            {
                tmp = z[0][j];
                idx = j;
            }
        }
        result.push_back(idx);
    }

    return result;
}


// 예측 값과 정답 값 비교하여 정확도 평가
float score(vector<int> result, vector<vector<float>> y)
{
    vector<int> resulty;

    for(int i=0; i<y.size(); i++)
    {
        float tmp = 0.0f;
        int idx = 0;    
        for(int j=0;j<y[i].size();j++)
        {
            if(tmp < y[i][j])
            {
                tmp = y[i][j];
                idx = j;
            }
        }
        resulty.push_back(idx);
    }

    int cnt = 0;
    for(int i=0; i<result.size(); i++)
    {
        if(result[i] == resulty[i])
        {
            cnt++;
        }
    }

    float score = (float) cnt/result.size();
    return score;
}


int main()
{
    chrono::steady_clock::time_point begin, data_end, score_end;

    // dataset 읽어오기
    begin = chrono::steady_clock::now();
    ifstream readFile;
    readFile.open("train.csv");

    int idx=0;

    if(readFile.is_open())    //파일이 열렸는지 확인
    {
        // 맨 윗줄 제거
        string row;
        getline(readFile, row);

        cout << "start load train data ...\n";

        while(!readFile.eof())    //파일 끝까지 읽었는지 확인
        {
            getline(readFile, row);
            istringstream ss(row);

            string num;
            
            while(getline(ss, num, ','))
            {
                float a = stof(num);
                load_data[idx].push_back(a);
            }
            
            idx++;
        }
        cout << "train_row: " << load_data.size() << ", train_column: " << load_data[0].size() << "\n";
        cout << "finish load train_data !\n";
        readFile.close();
    }
    else
        cout << "Can not open file\n";

    // X_train, y_train 만들기
    y_train.clear();

    for(int i=0; i<26880; i++)
    {
        for(int j=1; j<load_data[i].size(); j++)
        {
            X_train[i].push_back(load_data[i][j]);
        }
        float a = load_data[i][0];
        y_train.push_back(a);
    }


    //val 파일 읽기
    load_data.clear();
    load_data.resize(8400);

    readFile.open("val.csv");

    idx=0;

    if(readFile.is_open())
    {
        // 맨 윗줄 제거
        string row;
        getline(readFile, row);

        cout << "start load test data ...\n";

        while(!readFile.eof())
        {
            getline(readFile, row);
            istringstream ss(row);

            string num;
            
            while(getline(ss, num, ','))
            {
                float a = stof(num);
                load_data[idx].push_back(a);
            }
            
            idx++;
        }
        cout << "val_row: " << load_data.size() << ", val_column: " << load_data[0].size() << "\n";
        cout << "finish load val_data !\n";
        readFile.close();
    }
    else
        cout << "Can not open file\n";

    
    x_val.resize(8400);
    for(int i=0; i<8400; i++)
    {
        for(int j=1; j<load_data[i].size(); j++)
           x_val[i].push_back(load_data[i][j]);

        float a = load_data[i][0];
        y_val.push_back(a);
    }

    // test 데이터 불러오기
    load_data.clear();
    load_data.resize(6720);

    readFile.open("test.csv");

    idx=0;

    if(readFile.is_open())
    {
        // 맨 윗줄 제거
        string row;
        getline(readFile, row);

        cout << "start load train data ...\n";

        while(!readFile.eof())
        {
            getline(readFile, row);
            istringstream ss(row);

            string num;
            
            while(getline(ss, num, ','))
            {
                float a = stof(num);
                load_data[idx].push_back(a);
            }
            
            idx++;
        }
        cout << "test_row: " << load_data.size() << ", test_column: " << load_data[0].size() << "\n";
        cout << "finish load test_data !\n";
        readFile.close();
    }
    else
        cout << "Can not open file\n";


    for(int i=0; i<6720; i++)
    {
        for(int j=1; j<load_data[i].size(); j++)
        {
            x_test[i].push_back(load_data[i][j]);
        }

        float a = load_data[i][0];
        y_test.push_back(a);
    }

    data_end = chrono::steady_clock::now();
    cout << "Data reading time: " << (chrono::duration_cast<chrono::microseconds>(data_end-begin).count())/1000000.0f << endl;


    
    kaiming_init(w1, w1.size());
    kaiming_init(w2, w2.size());

    
    vector<vector<float>> y_train_enc = y_train_encoded(y_train);
    vector<vector<float>> y_val_enc = y_train_encoded(y_val);
    vector<vector<float>> y_test_enc = y_train_encoded(y_test);

    fit(x_val, y_val_enc, X_train, y_train_enc, 100);


    vector<int> result;
    result = predict(x_test, w1, w2);

    float plz = score(result, y_test_enc);
    cout << "score : " << plz << endl;   

    score_end = chrono::steady_clock::now();

    cout << endl;
    cout << "[ NN with GPU ]" << endl;
    cout << "* train dataset\t\t: 26880" << endl;
    cout << "* test dataset\t\t: 6720" << endl;
    cout << "* validation dataset\t: 8400" << endl;
    cout << endl;
    cout << "* Data reading time\t: " << (chrono::duration_cast<chrono::microseconds>(data_end-begin).count())/1000000.0f << endl;
    cout << "* epoch\t\t\t: 100" << endl;
    cout << "* Score\t\t\t: " << plz << endl;
    cout << "* Total time\t\t: " << (chrono::duration_cast<chrono::microseconds>(score_end-begin).count())/1000000.0f << endl;
    
    return 0;
}