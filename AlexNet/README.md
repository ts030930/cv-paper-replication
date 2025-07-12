# ImageNet Classification with Deep Convolutional Neural Networks

## Architecture


<img width="1460" height="449" alt="image" src="https://github.com/user-attachments/assets/e577e8e6-b786-4346-83b3-955be51275cf" />



Input Layer : (3,224,224)

Conv 1 Layer : kernel =  (11,11), out_channel = 96, stride = 4, padding = 2

LRN + Relu Layer

Max pooling Layer : kernel =  (3,3), stride = 2 

Conv 2 Layer : kernel =  (5,5), out_channel = 256, stride = 1, padding = 2

LRN + Relu Layer

Max pooling Layer : kernel =  (3,3), stride = 2 

Conv 3 Layer : kernel =  (3,3), out_channel = 384, stride = 1, padding = 1

Relu Layer

Conv 4 Layer : kernel =  (3,3), out_channel = 384, stride = 1, padding = 1

Relu Layer

Conv 5 Layer : kernel =  (3,3), out_channel = 256, stride = 1, padding = 1

Relu Layer

Max Pooling Layer : kernel =  (3,3), stride = 2 

Dropout Layer

Fc 1 Layer : in_features = 6*6*256 (Apply Max Pooling on 13*13*256), out_features = 4096

Dropout Layer

Fc 2 Layer : in_features = 4096, out_features = 4096

Fc 3 Layer : in_features = 4096, out_features = 1000

Softmax Layer 


## Overlapping Pooling

본 논문에서 Pooling으로 Max Overlapping Pooling을 사용했는데, stride = 2로 설정하고 kernel = 3으로 설정하여 3x3 커널보다 작은 stride로 이동하게 함으로써
Overlapping 시켰다. 해당 부분이 Overlapping 됨으로써 더 많은 정보를 반영할 수 있게되고 성능이 향상된다고 보고했다.
논문에서는 Overfitting을 억제하는 효과가 조금 있음을 발견했다고 한다.

## Local Response Normalization (LRN)


<img width="820" height="188" alt="image" src="https://github.com/user-attachments/assets/ffe2f6b9-bf8e-4a68-9b39-8997750374fa" />


이 방식은 정규화 기법의 일종으로
Relu 사용시 양수의 값이 매우 크다면 해당 값이 너무 큰 비중을 차지 한 상태로 학습을 지속시키게 된다. ( Relu의 함수적 특성에 의해서)
그러므로 각 채널(i)에서 주변 채널의 해당 위치 값들 (j)를 제곱하여 더 함으로써 정규화하는 방식이다.
즉 너무 큰 값을 제어할 수 있는 방식으로 현재에 와서는 잘 쓰이지 않는 방식이다. 
BN은 전체의 분포를 정규화함으로써 더 좋은 결과를 낸다는것이 실험적으로 입증됬다.










