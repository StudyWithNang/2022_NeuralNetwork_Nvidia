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

vector<vector<float>> load_data(42000);
vector<vector<float>> X_train(42000);
vector<float> y_train;
vector<float> losses, val_losses;
vector<vector<float>> w1(784), w2(784);
vector<float> b1(10), b2(10);
vector<vector<float>> output(10), a1(10);
vector<float> b1_grad(10);
float b2_grad;
vector<vector<float>> w1_grad(784), w2_grad(784);
    

float l1 = 0, l2 = 0;

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


// void relu(vector<float> a, vector<float>& out){
//     for(int i=0; i<a.size(); i++)
//     {
//         out[i] = (0 < a[i]) ? a[i] : 0; // max ~
//     }
// }

void sigmoid(vector<vector<float>> a, vector<vector<float>>& out){
    for(int i=0; i<a.size(); i++)
    {
        for(int j=0; j<a[i].size(); j++)
        {
            out[i][j] = 1/(1+exp(-a[i][j]));
        }
    }
}


vector<vector<float>> forward(vector<vector<float>> x, vector<vector<float>> &w1, vector<vector<float>> &w2, vector<vector<float>> &a1){
    vector<vector<float>> out;
    out = dot(x, w1);  //(1,10) w1(784,10)
    sumb(out, b1, out); //(1, 10)
    sigmoid(out, a1); //(1,10)

    out = dot(a1, w2); //w2:(10,10), out:(1,10)
    sumb(out, b2, out);
    //summ(out, b2, out); //b2,out: (1,10)

    return out;  //(1,10)
}

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

vector<vector<float>> y_train_encoded(vector<float> &y){
    vector<vector<float>> output(y.size(), vector<float>(10, 0));

    for(int i=0;i<output.size();i++)
    {
        output[i][int(y[i])] = 1;
    }
    return output;
}

void backprop(vector<vector<float>> x, vector<vector<float>> err, vector<vector<float>> &w2, vector<vector<float>> &w1_grad, vector<vector<float>> &w2_grad, vector<float> &b1_grad, float &b2_grad){
    for(int i=0; i<err.size(); i++)
    {
        b2_grad = 0.0f;
        for(int j=0; j<err[i].size(); j++)
        {
            b2_grad += err[i][j];
        }
    }

    vector<vector<float>> a1T = transpose(a1);
    w2_grad = dot(a1T, err);

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

vector<vector<float>> training(vector<vector<float>> x, vector<vector<float>> y, vector<vector<float>> &w1, \
                    vector<vector<float>> &w2, vector<vector<float>> &w1_grad, vector<vector<float>> &w2_grad, \
                    vector<float> &b1, vector<float> &b2, vector<float> &b1_grad, \
                    float &b2_grad, vector<vector<float>> &a1)
{
    vector<vector<float>> z = forward(x, w1, w2, a1);
    vector<vector<float>> a = softmax(z);

    vector<vector<float>> err(y.size());

    for(int i=0; i<y.size(); i++)
    {
        for(int j=0; j<y[i].size(); j++)
        {
            err[i].push_back(-(y[i][j]-a[i][j]));
        }
    }

    backprop(x, err, w2, w1_grad, w2_grad, b1_grad, b2_grad);

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

float reg_loss(vector<vector<float>> w1, vector<vector<float>> w2)
{
    float tmp1 = 0.0f;

    for(int i=0; i<w1.size(); i++)
    {
        for(int j=0; j<w1[i].size(); j++)
        {
            if(w1[i][j] < 0) w1[i][j] = -w1[i][j];
            if(w2[i][j] < 0) w2[i][j] = -w2[i][j];
            
            tmp1 += l1*(w1[i][j]) + w2[i][j] + l2/2*pow(w1[i][j],2) + pow(w2[i][j],2);
        }
    }
    return tmp1;
}

void update_val_loss(vector<vector<float>> x_val, vector<vector<float>> y_val, vector<vector<float>> w1, vector<vector<float>> w2, vector<float> b1, vector<float> b2, vector<vector<float>> a1)
{
    vector<vector<float>> smallx(1, vector<float>(784, 0)), smally(1, vector<float>(10, 0));
    for(int i=0; i<x_val.size(); i++)
    {
        //smallx = x_val[i];
        copy(x_val[i].begin(), x_val[i].end(), smallx[0].begin());
        copy(y_val[i].begin(), y_val[i].end(), smally[0].begin());

        vector<vector<float>> z = forward(smallx, w1, w2, a1);
        vector<vector<float>> a = softmax(z);
        float val_loss = 0.0f;
        for(int ii=0;ii<a.size();ii++)
        {
            for(int jj=0;jj<a[ii].size();jj++)
            {
                val_loss += (-y_val[ii][jj] * log(a[ii][jj]));
            }
        }
        val_losses.push_back((val_loss+reg_loss(w1, w2))/smally.size());
    }
}


void fit(vector<vector<float>> x, vector<vector<float>> y, int epochs){
    vector<vector<float>> trained_a(10);
    vector<float> loss(trained_a.size());
    for(int i=0; i<epochs; i++)
    {
        float loss = 0.0f;
        cout << ".";

        vector<vector<float>> smallx(1, vector<float>(784, 0.0)), smally(1, vector<float>(10, 0));
        for(int i=0; i<x_val.size(); i++)
        {
            copy(x_val[i].begin(), x_val[i].end(), smallx[0].begin());
            copy(y_val[i].begin(), y_val[i].end(), smally[0].begin());
            trained_a = training(smallx, smally, w1, w2,w1_grad, w2_grad, b1, b2, b1_grad, b2_grad, a1);

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
        update_val_loss(x_val, y_val, w1, w2, b1, b2, a1);
    }
    cout << endl;
}

// n_in=784, n_out=10
void kaiming_init(vector<vector<float>> &w, int n_in){
    float std = sqrt(2/(float) n_in);
    
    random_device rd;
    mt19937 gen(rd()); 
    normal_distribution<float> dist(0.0f, std); 

    for (int i=0; i<n_in; i++){
        for(int j=0; j<10; j++)
        {
            //cout << "(" << i << ", " << j << ") " << dist(gen) << "\t";
            w[i].push_back(dist(gen));
        }
    }
}

int main()
{
    chrono::steady_clock::time_point begin, end;

    // 데이터 읽기 !
    begin = chrono::steady_clock::now();
    ifstream readFile;             //읽을 목적의 파일 선언
    readFile.open("train.csv");    //파일 열기

    int idx=0;

    if(readFile.is_open())    //파일이 열렸는지 확인
    {
        // 맨 윗줄 제거
        string row;
        getline(readFile, row);

        cout << "start load data ...\n";

        //cout << "**************" << idx << "***************\n";
        getline(readFile, row);
        istringstream ss(row);

        string num;
        
        while(getline(ss, num, ','))
        {
            //int a = atoi(num.c_str());
            float a = stof(num);
            load_data[idx].push_back(a);
        }
        
        idx++;
        cout << "row: " << load_data.size() << ", column: " << load_data[0].size() << "\n";
        cout << "finish load data !\n";
        readFile.close();
    }
    else
        cout << "Can not open file\n";

    
    // X_train, y_train 만들기
    y_train.clear();
    for(int i=0; i<1; i++)
    {
        for(int j=1; j<load_data[i].size(); j++)
           X_train[i].push_back(load_data[i][j]);

        int a = load_data[i][0];
        y_train.push_back(a);        
    }
    end = chrono::steady_clock::now();
    cout << "Data reading time: " << (chrono::duration_cast<chrono::microseconds>(end-begin).count())/1000000.0f << endl;


    cout << "[ x ]\n";
    for(int i=0; i<X_train[0].size(); i++)
        cout << X_train[0][i] << " ";
    
    kaiming_init(w1, 784);
    kaiming_init(w2, 784);

    
    vector<vector<float>> y_train_enc = y_train_encoded(y_train);
    fit(X_train, y_train_enc, 2);

}
