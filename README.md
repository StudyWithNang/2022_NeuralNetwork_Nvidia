# 2022_NeuralNetwork_Nvidia

##### 병렬분산 프로젝트로서 **C++** 로 코드를 구현하여 **CPU**와 **GPU**에서 시간을 비교

### 주제: 선형함수로 이루어진 신경망으로 이미지 분류  
### Model architecture
### : layer1 + sigmoid + layer2
![image](https://user-images.githubusercontent.com/90039228/209432256-b02ba9aa-9bc3-435a-9930-f9cba46a9ed3.png)

### Dataset: MNIST  
(본 프로젝트에서 사용한 데이터는 train data: 26880개, validation data: 8400개, test data: 6720개로 <b>총 42000개</b>)

## 프로젝트 개요 및 필요성
+ 신경망 학습 시 처리해야할 데이터 양이 많을수록 학습 속도가 느려지고 메모리가 부족  
+ 대부분의 딥러닝 코드는 Python으로 구현되지만 Python은 다른 프로그래밍 언어에 비해 속도가 느리기 때문에 실시간 데이터에 대응하기 어려움  
(속도의 문제점을 개선하기 위해 실제 어플리케이션에서는 주로 C/C++를 사용)
+ 실제 사회에서도 GPU 사용 부분에 대한 고민이 필수적    

<b>⇒ 이미지 분류를 위한 학습 과정에 많은 연산이 존재</b>  
<b>⇒ 이에 대한 고민을 통해 CPU와 GPU에 따른 신경망 Neural Network 학습 속도를 비교함</b>

## Flowchart
![image](https://user-images.githubusercontent.com/90039228/208378832-ab22bf29-8f63-42f8-acd8-c462e5c0e293.png)


## CPU version 실행
```
$ g++ -o main_cpu main_cpu.cpp
$ ./main_cpu
```

## GPU version 실행
```
$ nvidia -o main_gpu main_cpu.cpp
$ ./main_gpu
```

## 실행 결과
### epoch : 100
![image](https://user-images.githubusercontent.com/90039228/209431761-648717a0-2599-45a8-a2ec-c354c7f7bf0f.png)
  * 데이터셋을 읽어오는 시간은 CPU, GPU가 각각 약 12초, 1초로 GPU가 약 12배 빠름
  * 하지만 전체 시간은 각각 약 77분, 60분으로 17분 차이
 
### epoch : 200
![image](https://user-images.githubusercontent.com/90039228/209431819-3eb76773-859d-4341-9245-d28c0f399d3c.png)
  * 데이터셋을 읽어오는 시간은 CPU, GPU가 각각 약 13초, 1초로 GPU가 약 12배 빠름
  * 하지만 전체 시간은 각각 약 156분, 122분으로 34분 차이
  
## 2022/12/16 complete
