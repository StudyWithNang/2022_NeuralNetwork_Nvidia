#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>

using namespace std;

vector<vector<float>> x_train = {{1, 4, 6}, {2, 3, 1}}; //(2, 3)
vector<vector<float>> w1 = {{1, 2}, {2, 2}, {4, 4}}; //(3, 2)
vector<vector<float>> w2 = {{3, 1}, {4, 6}}; //(2, 2)
vector<float> b1 = {0}; //(1, 2)
vector<float> b2 = {0}; //(1, 2)
vector<vector<float>> a1(1, vector<float>(2, 0));

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


void summ(vector<vector<float>> a, vector<vector<float>> b, vector<vector<float>>& out){   
    for(int i=0; i<a.size(); i++)
    {
        for(int j=0; j<a[i].size(); j++)
        {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
}

void sumb(vector<vector<float>> a, vector<float> b, vector<vector<float>>& out)
{
    for(int i=0;i<a.size();i++)
    {
        for(int j=0;j<a[i].size();j++)
        {
            out[i][j] = a[i][j] + b[j];
        }
    }
}

void sigmoid(vector<vector<float>> a, vector<vector<float>>& out){
    for(int i=0; i<a.size(); i++)
    {
        for(int j=0; j<a[i].size(); j++)
        {
            out[i][j] = 1.0/((float)1.0+exp(-a[i][j]));
        }
    }
}

vector<vector<float>> forward(vector<vector<float>> &x, vector<vector<float>> &w1, vector<vector<float>> &w2, vector<vector<float>> &a1){
    vector<vector<float>> out;
    out = dot(x, w1);  //(1,10) w1(784,10)
    sumb(out, b1, out); //(1, 10)
    sigmoid(out, a1); //(1,10)
    out = dot(a1, w2); //w2:(10,10), out:(1,10)
    sumb(out, b2, out);
    //summ(out, b2, out); //b2,out: (1,10)
    return out;  //(1,10)
}

int main()
{
    vector<vector<float>> smallx(1, vector<float>(784, 0)), smally(1, vector<float>(10, 0));
    for(int i=0; i<x_train.size(); i++)
    {
        //smallx = x_val[i];
        smallx[0].assign(x_train[i].begin(), x_train[i].end());
        vector<vector<float>> z = forward(smallx, w1, w2, a1);

        for(int i=0; i<z.size(); i++)
        {
            for(int j=0; j<z[i].size(); j++)
            {
                cout << z[i][j] << " ";
            }
            cout << endl;
        }
    }
    return 0;
}