# 2022_NeuralNetwork_Nvidia

병렬분산 프로젝트로서 **C++** 로 코드를 구현하여 **CPU**와 **GPU**에서 시간을 비교하였습니다.

주제: 선형함수로 이루어진 신경망으로 이미지 분류  
Model architecture: layer1 + sigmoid + layer2

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

