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


vector<vector<float>> sigmoid(vector<vector<float>> a){
    vector<vector<float>> out(a.size(), vector<float>(a[0].size(), 0));
    for(int i=0; i<a.size(); i++)
    {
        for(int j=0; j<a[i].size(); j++)
        {
            float tmp = a[i][j] / 4000;
            out[i][j] = 1.0/((float)1.0+exp(-tmp));
        }
    }
    return out;
}


vector<vector<float>> forward(vector<vector<float>> &x, vector<vector<float>> &w1, vector<vector<float>> &w2, vector<vector<float>> &a1){
    vector<vector<float>> out;
    out = dot(x, w1);
    sumb(out, b1, out); //(1, 10)
    //cout << out[0][0] << endl;
    a1 = sigmoid(out); //(1,10)
    out = dot(a1, w2); //w2:(10,10), out:(1,10)
    sumb(out, b2, out);
    //summ(out, b2, out); //b2,out: (1,10)
    return out;  //(1,10)
}

vector<vector<float>> test_forward(vector<vector<float>> &x, vector<vector<float>> &w1, vector<vector<float>> &w2){
    vector<vector<float>> out;
    out = dot(x, w1);  //(1,10) w1(784,10)
    sumb(out, b1, out); //(1, 10)
    out = sigmoid(out); //(1,10)
    cout << out[0][0] << endl;
    out = dot(out, w2); //w2:(10,10), out:(1,10)
    sumb(out, b2, out);
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
    w2_grad = dot(a1T, err);

    vector<vector<float>> w2T = transpose(w2);
    vector<vector<float>> err_to_hidden = dot(err, w2T); //err_to_hidden(1, 10)

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
    // for(int j=0; j<10; j++)
    // {
    //     cout << z[0][j] << " ";
    // }
    // cout << endl;

    vector<vector<float>> a = softmax(z);
    vector<vector<float>> err(y.size());

    for(int i=0; i<y.size(); i++)
    {
        for(int j=0; j<y[i].size(); j++)
        {
            err[i].push_back(-(y[i][j]-a[i][j]));
        }
    }

    backprop(x, err, w2, w1_grad, w2_grad, b1_grad, b2_grad, a1);

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

    for(int i=0; i<w1[0].size(); i++) //~10
    {
        for(int j=0; j<w1.size(); j++) //~784
        {
            if(w1[j][i] < 0) w1[j][i] = -w1[j][i];
            
            tmp1 += l1*(w1[j][i]) + l2/2*pow(w1[j][i],2);
            //tmp1 += l1*(w1[i][j]) + w2[i][j] + l2/2*pow(w1[i][j],2) + pow(w2[i][j],2);
        }

        for(int j=0; j<w2[i].size(); j++) //~10
        {
            if(w2[i][j] < 0) w2[i][j] = -w2[i][j];
            tmp1 += w2[i][j] + pow(w2[i][j],2);
        }
    }

    return tmp1;
}

void update_val_loss(vector<vector<float>> x_val, vector<vector<float>> y_val, vector<vector<float>> &w1, vector<vector<float>> &w2, \
                    vector<float> &b1, vector<float> &b2, vector<vector<float>> &a1)
{
    // vector<vector<float>> smallx(1, vector<float>(784, 0)), smally(1, vector<float>(10, 0));
    // cout << x_val.size() << " " << x_val[0].size() << endl;
    // for(int i=0; i<x_val.size(); i++)
    // {
    //     //smallx = x_val[i];
    //     smallx[0].assign(x_val[i].begin(), x_val[i].end());
    //     smally[0].assign(y_val[i].begin(), y_val[i].end());
    //     //copy(x_val[i].begin(), x_val[i].end(), smallx[0].begin());
    //     //copy(y_val[i].begin(), y_val[i].end(), smally[0].begin());

    //     vector<vector<float>> z = test_forward(smallx, w1, w2);
    //     // for(int j=0; j<z.size(); j++)
    //     // {
    //     //     for(int k=0; k<z[0].size(); k++)
    //     //     cout << z[0][j] << " ";
    //     // }
    //     // cout << endl;
    //     vector<vector<float>> a = softmax(z);
    //     float val_loss = 0.0f;
    //     for(int ii=0;ii<a.size();ii++)
    //     {
    //         for(int jj=0;jj<a[ii].size();jj++)
    //         {
    //             val_loss += (-y_val[ii][jj] * log(a[ii][jj]));
    //         }
    //     }
    //     val_losses.push_back((val_loss+reg_loss(w1, w2))/smally.size());
    // }



    vector<vector<float>> z = forward(x_val, w1, w2, a1);
    // for(int j=0; j<z.size(); j++)
    // {
    //     for(int k=0; k<z[0].size(); k++)
    //     cout << z[j][k] << " ";
    // }
    // cout << endl;
    vector<vector<float>> a = softmax(z);
    float val_loss = 0.0f;
    for(int ii=0;ii<a.size();ii++)
    {
        for(int jj=0;jj<a[ii].size();jj++)
        {
            val_loss += (-y_val[ii][jj] * log(a[ii][jj]));
        }
    }
    val_losses.push_back((val_loss+reg_loss(w1, w2))/y_val.size());


}


void fit(vector<vector<float>> &x_val, vector<vector<float>> &y_val, vector<vector<float>> &x, vector<vector<float>> &y, int epochs){
    vector<vector<float>> trained_a(10);
    vector<float> loss(trained_a.size());
    for(int i=0; i<epochs; i++)
    {
        float loss = 0.0f;
        cout << ".";

        vector<vector<float>> smallx(1, vector<float>(784, 0.0)), smally(1, vector<float>(10, 0));
        for(int j=0; j<x.size(); j++)
        {
            smallx[0].assign(x[j].begin(), x[j].end());
            smally[0].assign(y[j].begin(), y[j].end());
            // copy(x_val[i].begin(), x_val[i].end(), smallx[0].begin());
            // copy(y_val[i].begin(), y_val[i].end(), smally[0].begin());

            trained_a = training(smallx, smally, w1, w2, w1_grad, w2_grad, b1, b2, b1_grad, b2_grad, a1_train);

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

        for(int j=0;j<x_val.size();j++)
        {
            smallx[0].assign(x_val[j].begin(), x_val[j].end());
            smally[0].assign(y_val[j].begin(), y_val[j].end());
            update_val_loss(smallx, smally, w1, w2, b1, b2, a1_train);
        }
        // update_val_loss(x_val, y_val, w1, w2, b1, b2, a1_val);
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

vector<int> predict(vector<vector<float>> x, vector<vector<float>> &w1, vector<vector<float>> &w2){
    vector<int> result;
    vector<vector<float>> smallx(1);

    for(int i=0; i<x.size(); i++)
    {
        smallx[0].assign(x[i].begin(), x[i].end());
        //cout << smallx[0][347] << ' ';
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

        // for(int j=0; j<z[0].size(); j++)
        // {
        //     cout << z[0][j] << " ";
        // }
        // cout << endl;
    }

    // for(int i=0; i<result.size(); i++)
    // {
    //     cout << result[i] << endl;
    // }
    cout << "finish predict" << endl;
    return result;
}

float score(vector<int> result, vector<vector<float>> y)
{
    //vector<vector<float>> smally(1, vector<float>(784, 0.0));
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
        //smally[0].assign(y[i].begin(), y[i].end());
        // float tmp = 0.0f;
        // int idx = 0;
        // for(int j=0; j<smally[0].size(); j++)
        // {
        //     if(tmp < smally[0][j])
        //     {
        //         tmp = smally[0][j];
        //         idx = j;
        //     }
        // }
    }
    // for(int i=0;i<result.size();i++)
    //     cout << result[i] << ' ';

    int cnt = 0;
    for(int i=0; i<result.size(); i++)
    {
        if(result[i] == resulty[i])
        {
            cnt++;
        }
    }

    // for(int i=0; i<100; i++)
    // {
    //     cout << result[i] << " vs " << resulty[i] << endl;
    // }

    float score = (float) cnt/result.size();
    return score;
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

        cout << "start load train data ...\n";

        while(!readFile.eof())    //파일 끝까지 읽었는지 확인
        {
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

    readFile.open("val.csv");    //파일 열기

    idx=0;

    if(readFile.is_open())    //파일이 열렸는지 확인
    {
        // 맨 윗줄 제거
        string row;
        getline(readFile, row);

        cout << "start load test data ...\n";

        while(!readFile.eof())    //파일 끝까지 읽었는지 확인
        {
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
        }
        cout << "val_row: " << load_data.size() << ", val_column: " << load_data[0].size() << "\n";
        cout << "finish load val_data !\n";
        readFile.close();
    }
    else
        cout << "Can not open file\n";

    
    x_val.resize(8400);
    // X_train, y_train 만들기
    //y_train.clear();
    //cout << "x_val size: " << x_val.size() << " " << x_val[0].size() << " " << y_val.size() << " " << y_val[0].size() << " " << load_data[0].size();
    for(int i=0; i<8400; i++)
    {
        for(int j=1; j<load_data[i].size(); j++)
           x_val[i].push_back(load_data[i][j]);

        float a = load_data[i][0]; //꼭 int로 해야하나?
        y_val.push_back(a);
    }

    // test 데이터 불러오기
    load_data.clear();
    load_data.resize(6720);


    readFile.open("test.csv");    //파일 열기

    idx=0;

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
                //int a = atoi(num.c_str());
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
    //cout << "X_train size: " << X_train.size() << " " << X_train[0].size() << " " << y_train.size() << " " << load_data[0].size();


    end = chrono::steady_clock::now();
    cout << "Data reading time: " << (chrono::duration_cast<chrono::microseconds>(end-begin).count())/1000000.0f << endl;


    // cout << "[ x ]\n";
    // for(int i=0; i<X_train[0].size(); i++)
    //     cout << X_train[0][i] << " ";
    
    kaiming_init(w1, w1.size());
    kaiming_init(w2, w2.size());

    
    vector<vector<float>> y_train_enc = y_train_encoded(y_train);
    vector<vector<float>> y_val_enc = y_train_encoded(y_val);
    vector<vector<float>> y_test_enc = y_train_encoded(y_test);
    fit(x_val, y_val_enc, X_train, y_train_enc, 40);
    cout << "training end ~ " << endl;

    vector<int> result;
    result = predict(x_test, w1, w2);
    cout << "predict finish" << endl;

    // for(int i=0; i<result.size(); i++)
    // {
    //     cout << result[i] << endl;
    // }
    float plz = score(result, y_test_enc);
    cout << "score : " << plz << endl;
    cout << "the end ~ " << endl;

    return 0;
}
